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
import docx  # python-docx

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- Стили и боковая панель ---
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
    </style>
    <div class="center">
        <img src="https://www.maib.md/uploads/custom_blocks/image_1633004921_8nR1jw3Qfu_auto__0.png" width="300">
        <h1>HR-Recruiter</h1>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-title">Proiect: AI Recruiter pentru MAIB</div>', unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown("""
<div class="sidebar-text">
1. 📥 **Încărcarea posturilor vacante**  
2. 📄 **CV-ul utilizatorului**  
3. 🤖 **Căutarea posturilor potrivite**  
4. 🔍 **Analiza celei mai relevante poziții**  
5. ✅ **Acordul candidatului**  
6. 🗣️ **Primul interviu (general)**  
7. 💻 **Interviul tehnic**  
8. 📋 **Concluzia finală**  
</div>
""", unsafe_allow_html=True)

st.divider()

class DocumentChunk:
    def __init__(self, text, doc_name, page_num):
        self.text = text
        self.doc_name = doc_name
        self.page_num = page_num

class KnowledgeBase:
    
    def clear(self):
            self.chunks = []
            self.doc_texts = []
            self.uploaded_files = []


    
    def __init__(self):
        self.chunks = []
        self.uploaded_files = []
        self.doc_texts = []

    def split_text(self, text, max_chars=2000):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

    def load_pdf(self, file_content, file_name):
        if file_name in self.uploaded_files:
            return False
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
            st.error(f"Ошибка загрузки PDF: {e}")
            return False
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def load_docx(self, file_content, file_name):
        if file_name in self.uploaded_files:
            return False
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            doc = docx.Document(tmp_file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            text = "\n".join(full_text)
            for chunk in self.split_text(text):
                self.chunks.append(DocumentChunk(chunk, file_name, 0))
                self.doc_texts.append(chunk)
            self.uploaded_files.append(file_name)
            return True
        except Exception as e:
            st.error(f"Ошибка загрузки DOCX: {e}")
            return False
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def load_txt(self, file_content, file_name):
        if file_name in self.uploaded_files:
            return False
        try:
            text = file_content.decode('utf-8', errors='ignore')
            for chunk in self.split_text(text):
                self.chunks.append(DocumentChunk(chunk, file_name, 0))
                self.doc_texts.append(chunk)
            self.uploaded_files.append(file_name)
            return True
        except Exception as e:
            st.error(f"Ошибка загрузки TXT: {e}")
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
            st.warning(f"Формат файла {uploaded_file.name} не поддерживается.")
            return False

    def get_all_text(self):
        return "\n\n".join(self.doc_texts)

# Инициализация хранилища
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()

if 'vacancies_data' not in st.session_state:
    st.session_state.vacancies_data = []

##############################
# Загрузка вакансий с rabota.md для Moldova Agroindbank
headers = {'User-Agent': 'Mozilla/5.0'}
base_url = "https://www.rabota.md/ru/companies/moldova-agroindbank#vacancies"

if st.button("Încarcă ofertele de muncă de pe rabota.md"):
    with st.spinner("Загружаем вакансии..."):
        try:
            response = requests.get(base_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            links = soup.find_all('a', class_='vacancyShowPopup')
            urls = [urljoin(base_url, a['href']) for a in links]

            vacancies_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, url in enumerate(urls):
                resp = requests.get(url, headers=headers)
                resp.raise_for_status()
                soup_vac = BeautifulSoup(resp.text, 'html.parser')

                title_tag = soup_vac.find('h1')
                title = title_tag.get_text(strip=True) if title_tag else 'Не найдено'

                vacancy_content = soup_vac.find('div', class_='vacancy-content')
                description = vacancy_content.get_text(separator='\n', strip=True) if vacancy_content else 'Описание не найдено'

                vacancies_data.append({'url': url, 'title': title, 'description': description})

                status_text.text(f"[{i+1}/{len(urls)}] Сохранена вакансия: {title}")
                progress_bar.progress((i+1)/len(urls))
                time.sleep(0.1)

            st.session_state.vacancies_data = vacancies_data

        except Exception as e:
            st.error(f"Ошибка при загрузке вакансий: {e}")

# 🔄 Показываем список вакансий в сайдбаре, если они уже есть
if "vacancies_data" in st.session_state:
    with st.sidebar:
        st.markdown("### 🔎 Lista ofertelor MAIB:")
        st.success(f"Найдено вакансий: {len(st.session_state.vacancies_data)}")
        for vac in st.session_state.vacancies_data:
            st.markdown(
                f'<a href="{vac["url"]}" target="_blank" style="color:#40c1ac; text-decoration:none;">• {vac["title"]}</a>',
                unsafe_allow_html=True
            )


##############################
# Загрузка CV (PDF, DOCX, TXT)
st.markdown("### Încărcă CV-ul tău în format PDF, DOCX sau TXT")
uploaded_files = st.file_uploader("Файл с резюме", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

if uploaded_files:
    kb = st.session_state.get("knowledge_base", KnowledgeBase())
    kb.clear()  # 🧼 ОЧИСТКА ПЕРЕД НОВОЙ ЗАГРУЗКОЙ
    for uploaded_file in uploaded_files:
        kb.load_file(uploaded_file)
    st.session_state.knowledge_base = kb



if not st.session_state.knowledge_base.uploaded_files:
    st.info("Загрузите CV в формате PDF, DOCX или TXT для анализа")
    st.stop()

st.markdown("### 🔍 Cele mai relevante oferte pentru CV-ul încărcat:")

# Объединённый текст всех резюме
cv_text = st.session_state.knowledge_base.get_all_text()

# Все вакансии
vacancies = st.session_state.vacancies_data

# Если нет вакансий – остановим
if not vacancies:
    st.warning("Nu există oferte de muncă încărcate.")
    st.stop()

# Список описаний вакансий
vacancy_texts = [vac['description'] for vac in vacancies]

# Добавляем CV в список, чтобы сравнить его с каждым описанием вакансии
documents = [cv_text] + vacancy_texts

# TF-IDF + Косинусное сходство
vectorizer = TfidfVectorizer(stop_words='romanian')
tfidf_matrix = vectorizer.fit_transform(documents)
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Получаем индексы топ-3 вакансий
top_indices = similarity_scores.argsort()[::-1][:3]

for idx in top_indices:
    vac = vacancies[idx]
    score = similarity_scores[idx]
    st.markdown(f"#### 🏢 {vac['title']}")
    st.markdown(f"**URL:** [Deschide oferta]({vac['url']})")
    st.markdown(f"**Scor de relevanță:** `{score:.2f}`")
    with st.expander("📄 Descrierea ofertei"):
        st.write(vac['description'])

