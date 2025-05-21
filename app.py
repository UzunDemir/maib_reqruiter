import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json
from PyPDF2 import PdfReader
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import docx
from docx import Document
import io

from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- Stiluri și bara laterală ---
st.markdown("""
    <style>

        section[data-testid="stSidebar"] {
            background-color: #253646 !important;
        }
        .sidebar-title {
            color: white;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sidebar-text {
            color: white;
        }
        #MainMenu, footer, header {
            display: none !important;
        }
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            flex-direction: column;
            margin-top: 0vh;
        }
        .match-card {
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f0f2f6;
        }
        .match-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .progress-bar {
            height: 10px;
            background-color: #e0e0e0;
            border-radius: 5px;
            margin-top: 5px;
        }
        .progress-fill {
            height: 100%;
            border-radius: 5px;
            background-color: #40c1ac;
        }
    </style>
    <div class="center">
        <img src="https://www.maib.md/uploads/custom_blocks/image_1633004921_8nR1jw3Qfu_auto__0.png" width="300">
        <h1>AI HR-Recruiter</h1>
    </div>
""", unsafe_allow_html=True)

import streamlit as st

st.sidebar.markdown('<div class="sidebar-title">MAIB AI HR-Recruiter</div>', unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown("""
<div class="sidebar-text">

1. 📥 **Încărcarea posturilor vacante**  

   *Agentul încarcă automat toate posturile vacante actuale de la MAIB.*

2. 📄 **CV-ul utilizatorului**  

   *Utilizatorul își încarcă CV-ul pentru analiză.*

3. 🤖 **Căutarea posturilor potrivite**  

   *Agentul analizează CV-ul și identifică **top 3 posturi** relevante pentru experiența și competențele candidatului.*

4. 🔍 **Analiza celei mai relevante poziții**  

   * *Evidențiază **punctele forte** ale candidatului.*  
   * *Identifică **punctele slabe** sau lipsurile în competențe.*

5. ✅ **Acordul candidatului**  

   *Dacă este interesat, candidatul își exprimă acordul pentru a continua procesul.*

6. 🗣️ **Primul interviu (general)**  

   *Agentul pune întrebări generale, analizează răspunsurile și formulează **primele concluzii**.*

7. 💻 **Interviul tehnic**  

   *Evaluarea competențelor tehnice și furnizarea unui **feedback tehnic**.*

8. 📋 **Concluzia finală**  

   *Agentul oferă un verdict final: **recomandare pentru angajare** sau **refuz argumentat**.*

</div>
""", unsafe_allow_html=True)

st.divider()


class DocumentChunk:
    def __init__(self, text, doc_name, page_num):
        self.text = text
        self.doc_name = doc_name
        self.page_num = page_num

class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.uploaded_files = []
        self.doc_texts = []
    
    def clear(self):
        self.chunks = []
        self.doc_texts = []
        self.uploaded_files = []
    
    def split_text(self, text, max_chars=2000):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:  # Skip empty chunks
                chunks.append(chunk)
            start = end
        return chunks

    def load_pdf(self, file_content, file_name):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            with open(tmp_file_path, 'rb') as file:
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        for chunk in self.split_text(page_text):
                            self.chunks.append(DocumentChunk(chunk, file_name, page_num+1))
                            self.doc_texts.append(chunk)
            self.uploaded_files.append(file_name)
            return True
        except Exception as e:
            st.error(f"Eroare la încărcarea PDF: {str(e)}")
            return False
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def load_docx(self, file_content, file_name):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            doc = docx.Document(tmp_file_path)
            full_text = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:  # Skip empty paragraphs
                    full_text.append(text)
            text = "\n".join(full_text)
            for chunk in self.split_text(text):
                self.chunks.append(DocumentChunk(chunk, file_name, 0))
                self.doc_texts.append(chunk)
            self.uploaded_files.append(file_name)
            return True
        except Exception as e:
            st.error(f"Eroare la încărcarea DOCX: {str(e)}")
            return False
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def load_txt(self, file_content, file_name):
        try:
            text = file_content.decode('utf-8', errors='ignore')
            for chunk in self.split_text(text):
                self.chunks.append(DocumentChunk(chunk, file_name, 0))
                self.doc_texts.append(chunk)
            self.uploaded_files.append(file_name)
            return True
        except Exception as e:
            st.error(f"Eroare la încărcarea TXT: {str(e)}")
            return False

    def load_file(self, uploaded_file):
        name = uploaded_file.name.lower()
        content = uploaded_file.read()
        if name.endswith('.pdf'):
            return self.load_pdf(content, uploaded_file.name)
        elif name.endswith('.docx'):
            return self.load_docx(content, uploaded_file.name)
        elif name.endswith('.txt'):
            return self.load_txt(content, uploaded_file.name)
        else:
            st.warning(f"Formatul fișierului {uploaded_file.name} nu este suportat.")
            return False

    def get_all_text(self):
        return "\n\n".join(self.doc_texts)




# Inițializare baza de cunoștințe
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()

if 'vacancies_data' not in st.session_state:
    st.session_state.vacancies_data = []

##############################
# Încărcare oferte de pe rabota.md
def scrape_vacancy(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup_vac = BeautifulSoup(resp.text, 'html.parser')

        title_tag = soup_vac.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else 'Titlu indisponibil'

        vacancy_content = soup_vac.find('div', class_='vacancy-content')
        description = vacancy_content.get_text(separator='\n', strip=True) if vacancy_content else 'Descriere indisponibilă'

        return {'url': url, 'title': title, 'description': description}
    except Exception as e:
        st.error(f"Eroare la preluarea ofertei {url}: {str(e)}")
        return None

#######################################################################################################################
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_vacancies():
    base_url = "https://www.rabota.md/ru/companies/moldova-agroindbank#vacancies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    with st.spinner("Se încarcă ofertele de muncă..."):
        try:
            response = requests.get(base_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            links = soup.find_all('a', class_='vacancyShowPopup')
            urls = [urljoin(base_url, a['href']) for a in links]

            vacancies_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            total = len(urls)
            done = 0

            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(scrape_vacancy, url): url for url in urls}
                
                for future in as_completed(future_to_url):
                    result = future.result()
                    if result:
                        vacancies_data.append(result)
                        done += 1
                        progress_bar.progress(done / total)
                        status_text.text(f"[{done}/{total}] Ofertă încărcată: {result['title']}")

            st.session_state.vacancies_data = vacancies_data
            st.success(f"Au fost încărcate {len(vacancies_data)} oferte de muncă!")

        except Exception as e:
            st.error(f"Eroare la încărcarea ofertelor: {str(e)}")
##########################################################################################################################################





if st.button("Încarcă ofertele de muncă pentru tine..."):
    load_vacancies()

# Afișare lista oferte în bara laterală
if st.session_state.vacancies_data:
    with st.sidebar:
        st.markdown("### 🔎 Lista ofertelor MAIB:")
        st.success(f"Oferte găsite: {len(st.session_state.vacancies_data)}")
        for vac in st.session_state.vacancies_data:
            st.markdown(
                f'<a href="{vac["url"]}" target="_blank" style="color:#40c1ac; text-decoration:none;">• {vac["title"]}</a>',
                unsafe_allow_html=True
            )

##############################
# Încărcare CV
st.markdown("### 📄 Încărcă CV-ul tău (PDF, DOCX, TXT)")
uploaded_files = st.file_uploader("Selectează fișier(e)", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

if uploaded_files:
    kb = st.session_state.knowledge_base
    kb.clear()  # Curăță conținutul anterior
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        success = kb.load_file(uploaded_file)
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Se procesează fișierul {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
    
    st.session_state.knowledge_base = kb
    st.success(f"Au fost încărcate {len(uploaded_files)} fișiere!")

if not st.session_state.knowledge_base.uploaded_files:
    st.info("Te rugăm să încarci un CV pentru analiză")
    st.stop()

##############################
# Analiza potrivirilor
st.markdown("### 🔍 Cele mai relevante oferte pentru CV-ul tău")

cv_text = st.session_state.knowledge_base.get_all_text()
vacancies = st.session_state.vacancies_data

if not vacancies:
    st.warning("Nu există oferte de muncă disponibile. Te rugăm să încarci ofertele mai întâi.")
    st.stop()

# Procesare avansată a textelor
vacancy_texts = [f"{vac['title']}\n{vac['description']}" for vac in vacancies]
documents = [cv_text] + vacancy_texts

# TF-IDF îmbunătățit
vectorizer = TfidfVectorizer(
    stop_words=None,
    ngram_range=(1, 2),  # Include bigrame pentru mai mult context
    max_features=5000
)
##############################################################################


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests
import io
from docx import Document  # pip install python-docx



# --- Анализ совпадений вакансий ---
try:
    with st.spinner("Se analizează potrivirile..."):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    score_min, score_max = similarity_scores.min(), similarity_scores.max()
    if score_max - score_min > 0:
        normalized_scores = (similarity_scores - score_min) / (score_max - score_min) * 100
    else:
        normalized_scores = np.zeros_like(similarity_scores)

    normalized_scores = np.clip(normalized_scores, 0, 100)

    top_indices = similarity_scores.argsort()[::-1][:3]

    # Сохраняем top_indices в сессию, чтобы использовать дальше
    st.session_state.top_indices = top_indices

    for idx in top_indices:
        vac = vacancies[idx]
        score = normalized_scores[idx]

        with st.container():
            st.markdown(f"""
            <div class="match-card">
                <div class="match-header">
                    <h3><a href="{vac['url']}" target="_blank" style="text-decoration:none; color:inherit;">{vac['title']}</a></h3>
                    <h4>{score:.0f}% potrivire</h4>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {score}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("---")

except Exception as e:
    st.error(f"Eroare la analiza potrivirilor: {str(e)}")

# --- Работа с Deepseek API ---
api_key = st.secrets.get("DEEPSEEK_API_KEY")
if not api_key:
    st.error("API ключ не настроен. Пожалуйста, добавьте его в Secrets.")
    st.stop()

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Используем top_indices из сессии, если есть
top_indices = st.session_state.get("top_indices", [])
if len(top_indices) > 0:
    best_match_idx = top_indices[0]
    best_match_vacancy = vacancies[best_match_idx]

    # Отображаем кнопку для генерации анализа
    if st.button("🔍 Generează analiza de potrivire"):
        prompt = f"""
        Analizează corespondența dintre CV-ul candidatului și oferta de muncă.
        Mai întâi voi furniza CV-ul, apoi descrierea postului.

        CV-ul candidatului:
        {cv_text}
        
        Descrierea postului:
        {best_match_vacancy['description']}
        
        Vă rog să efectuați analiza conform următoarei structuri:

        1. Punctele forte ale CV-ului (potrivirea exactă cu cerințele postului)
        2. Punctele slabe sau lacunele din CV (unde candidatul nu corespunde)
        3. Recomandări concrete pentru îmbunătățirea CV-ului în vederea acestei poziții
        4. Procentajul general de potrivire (evaluat pe o scară de la 0 la 100%)
        5. Fiți cât mai concret, citați cerințele specifice din descrierea postului și punctele din CV.
        """

        try:
            with st.spinner("Generăm o analiză detaliată…"):
                data = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }

                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()

                result = response.json()
                analysis = result['choices'][0]['message']['content']

                # Сохраняем анализ в сессию, чтобы не пропадал
                st.session_state.analysis = analysis

        except Exception as e:
            st.error(f"Ошибка при запросе к API: {str(e)}")

# Показываем анализ, если он уже есть в сессии
if 'analysis' in st.session_state and st.session_state.analysis:
    st.markdown("## 📊 Analiză detaliată a conformității")
    st.markdown(st.session_state.analysis)

    # Кнопка для скачивания анализа в docx
    def create_word_document(text):
        doc = Document()
        doc.add_heading('Analiză detaliată a conformității', 0)
        for line in text.split('\n'):
            if line.strip():
                if line.startswith('##'):
                    doc.add_heading(line.replace('##', '').strip(), level=1)
                elif line.startswith('#'):
                    doc.add_heading(line.replace('#', '').strip(), level=2)
                else:
                    doc.add_paragraph(line)
        return doc

    doc = create_word_document(st.session_state.analysis)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)

    st.download_button(
        label="💾 Descarcă analiza (DOCX)",
        data=bio,
        file_name="analiza_potrivire.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
else:
    st.info("Apăsați butonul pentru a genera analiza de potrivire.")

# --- Интервью ---

def generate_interview_questions(cv_text):
    prompt = f"""
    Generează 10 întrebări pentru un interviu introductiv pe baza acestui CV:
    {cv_text}

    Cerințe:

    1. 3 întrebări despre experiența profesională
    2. 2 întrebări despre abilitățile tehnice
    3. 1 întrebare despre punctele slabe
    4. 1 întrebare despre motivație
    5. 1 întrebare despre așteptările salariale
    6. 2 întrebări biografice

    Întrebările trebuie să fie specifice și legate de CV

    Returnează doar o listă numerotată de întrebări, fără explicații suplimentare.
    """

    response = requests.post(
        url,
        headers=headers,
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
    )
    return response.json()['choices'][0]['message']['content']

def generate_candidate_profile(questions, answers):
    prompt = f"""
    Pe baza acestor întrebări și răspunsuri, creează un profil al candidatului:

    Întrebări:
    {questions}

    Răspunsuri:

    {answers}

    Структура профиля:
    ### 🧑‍💻 Portret profesional
    - Competențe principale
    - Experiență relevantă
    - Expertiză tehnică

    ### 🎯 Motivație și obiective
    - Interese profesionale
    - Așteptări de la job

    ### 📈 Puncte forte
    - Avantaje cheie
    - Competențe unice

    ### ⚠️ Zone de dezvoltare
    - Puncte slabe
    - Competențe de îmbunătățit

    ### 💰 Așteptări privind compensația
    - Așteptări salariale
    - Disponibilitate pentru negociere
    """

    response = requests.post(
        url,
        headers=headers,
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
    )
    return response.json()['choices'][0]['message']['content']

st.title("🤖 AI HR-Recruiter: Interviu introductiv")

# Инициализация состояния интервью
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
    st.session_state.questions = None
    st.session_state.answers = {}
    st.session_state.profile = None

if not st.session_state.interview_started:
    if st.button("🎤 A trece interviul introductiv", type="primary"):
        with st.spinner("Pregătim întrebările..."):
            st.session_state.questions = generate_interview_questions(documents[0])
            st.session_state.interview_started = True
        st.rerun()
        

if st.session_state.interview_started:
    st.success("Interviul a început! Vă rog să răspundeți la întrebările de mai jos.")

    questions_list = [q for q in st.session_state.questions.split('\n') if q.strip()]
    for i, question in enumerate(questions_list[:10]):
        st.session_state.answers[i] = st.text_area(
            label=f"**{i+1}:** {question}",
            value=st.session_state.answers.get(i, ""),
            key=f"answer_{i}"
        )

    if st.button("✅ Interviul s-a încheiat", type="primary"):
        with st.spinner("Analizăm răspunsurile..."):
            formatted_answers = "\n".join(
                [f"{i+1}. {q}\n   Ответ: {st.session_state.answers[i]}"
                 for i, q in enumerate(questions_list[:10])]
            )

            st.session_state.profile = generate_candidate_profile(
                st.session_state.questions,
                formatted_answers
            )

        st.success("Interviul s-a încheiat!")
        st.balloons()
        # Чтобы профиль сразу показался после окончания, перезапускаем
        st.rerun()

    

    st.success("Interviul s-a încheiat!")
    st.balloons()
    st.rerun()


      

if st.session_state.profile:
    st.markdown("## 📌 Profilul candidatului")
    st.markdown(st.session_state.profile)

    # Создание и скачивание DOCX профиля кандидата
    def create_word_document_profile(profile_text):
        doc = Document()
        doc.add_heading('Profil Candidat', 0)
        for line in profile_text.split('\n'):
            if line.strip():
                if line.startswith('###'):
                    doc.add_heading(line.replace('###', '').strip(), level=2)
                else:
                    doc.add_paragraph(line)
        return doc

    doc_profile = create_word_document_profile(st.session_state.profile)
    bio_profile = io.BytesIO()
    doc_profile.save(bio_profile)
    bio_profile.seek(0)

    st.download_button(
        label="💾 Descarcă profilul candidatului (DOCX)",
        data=bio_profile,
        file_name="profil_candidat.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

