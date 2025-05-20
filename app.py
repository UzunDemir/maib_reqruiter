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
        <h1>HR-Recruiter MAIB</h1>
    </div>
""", unsafe_allow_html=True)

import streamlit as st

st.sidebar.markdown('<div class="sidebar-title">Proiect: AI Recruiter pentru MAIB</div>', unsafe_allow_html=True)
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

            # Use threading to speed up scraping
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(scrape_vacancy, urls))
            
            for i, result in enumerate(results):
                if result:
                    vacancies_data.append(result)
                    progress = (i + 1) / len(urls)
                    progress_bar.progress(progress)
                    status_text.text(f"[{i+1}/{len(urls)}] Ofertă încărcată: {result['title']}")

            st.session_state.vacancies_data = [v for v in vacancies_data if v is not None]
            st.success(f"Au fost încărcate {len(st.session_state.vacancies_data)} oferte de muncă!")
            
        except Exception as e:
            st.error(f"Eroare la încărcarea ofertelor: {str(e)}")

if st.button("Încarcă ofertele de muncă de pe rabota.md"):
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
st.markdown("### 📄 Încărcă CV-ul tău (PDF, DOCX sau TXT)")
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

# try:
#     with st.spinner("Se analizează potrivirile..."):
#         tfidf_matrix = vectorizer.fit_transform(documents)
#         similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
#     # Normalizează scorurile între 0 și 100 pentru o afișare mai bună
#     normalized_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min()) * 100
#     normalized_scores = np.clip(normalized_scores, 0, 100)
    
#     # Obține top 3 oferte
#     top_indices = similarity_scores.argsort()[::-1][:3]
    
#     for idx in top_indices:
#         vac = vacancies[idx]
#         score = normalized_scores[idx]
        
#         with st.container():
#             st.markdown(f"""
#             <div class="match-card">
#                 <div class="match-header">
#                     <h3>{vac['title']}</h3>
#                     <h4>{score:.0f}% potrivire</h4>
#                 </div>
#                 <div class="progress-bar">
#                     <div class="progress-fill" style="width: {score}%"></div>
#                 </div>
#                 <p><a href="{vac['url']}" target="_blank">🔗 Vezi oferta completă</a></p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             with st.expander("📝 Detalii despre ofertă"):
#                 st.write(vac['description'])
                
#             st.write("---")

# except Exception as e:
#     st.error(f"Eroare la analiza potrivirilor: {str(e)}")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

try:
    with st.spinner("Se analizează potrivirile..."):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Защита от деления на 0
    score_min, score_max = similarity_scores.min(), similarity_scores.max()
    if score_max - score_min > 0:
        normalized_scores = (similarity_scores - score_min) / (score_max - score_min) * 100
    else:
        normalized_scores = np.zeros_like(similarity_scores)

    normalized_scores = np.clip(normalized_scores, 0, 100)

    top_indices = similarity_scores.argsort()[::-1][:3]

    for idx in top_indices:
        vac = vacancies[idx]
        score = normalized_scores[idx]

        with st.container():
            st.markdown(f"""
            <div class="match-card">
                <div class="match-header">
                    <h3>{vac['title']}</h3>
                    <h4>{score:.0f}% potrivire</h4>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {score}%"></div>
                </div>
                <p><a href="{vac['url']}" target="_blank">🔗 Vezi oferta completă</a></p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📝 Detalii despre ofertă"):
                st.write(vac['description'])

            st.write("---")

except Exception as e:
    st.error(f"Eroare la analiza potrivirilor: {str(e)}")

    import streamlit as st
import requests
import json

# Получение API ключа
api_key = st.secrets.get("DEEPSEEK_API_KEY")
if not api_key:
    st.error("API ключ не настроен. Пожалуйста, добавьте его в Secrets.")
    st.stop()

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Ввод данных пользователем
st.title("Анализ резюме рекрутером")

cv_text = st.text_area("Вставьте текст резюме (CV):", height=300)
target_profession = st.text_input("Укажите релевантную профессию:")

if st.button("Проанализировать CV"):
    if not cv_text or not target_profession:
        st.error("Пожалуйста, заполните оба поля — текст резюме и релевантную профессию.")
    else:
        # Формируем промпт
        prompt = (
            f"Ты — опытный рекрутер. Твоя задача — проанализировать резюме кандидата для позиции '{target_profession}'.\n\n"
            "Вот текст резюме:\n"
            f"{cv_text}\n\n"
            "Пожалуйста, выдели:\n"
            "1. Сильные стороны кандидата с точки зрения этой профессии.\n"
            "2. Слабые стороны и пробелы в резюме относительно данной вакансии.\n"
            "3. Рекомендации, как улучшить резюме или какие навыки развивать.\n"
            "4. Краткий вывод — стоит ли рассматривать кандидата для этой позиции.\n"
            "Выведи ответ структурированным и четким списком."
        )

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Ты — профессиональный рекрутер с опытом анализа резюме."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            # Получаем ответ модели
            reply = result['choices'][0]['message']['content']
            st.markdown("### Анализ резюме:")
            st.write(reply)
        else:
            st.error(f"Ошибка API: {response.status_code} — {response.text}")


