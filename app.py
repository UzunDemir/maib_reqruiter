import os
import streamlit as st
import requests
import json
import time
from PyPDF2 import PdfReader
import tempfile
from datetime import datetime
from transformers import AutoTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json


st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# Инициализация токенизатора
#tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm")
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")



# Настройка логотипа
# st.sidebar.image("https://www.maib.md/uploads/custom_blocks/image_1633004921_8nR1jw3Qfu_auto__0.png", use_container_width=True)

# Кастомизация через HTML+JS для боковой панели + скрытие верхнего меню, футера и хедера
st.markdown("""
    <style>
        /* Сайдбар целиком */
        section[data-testid="stSidebar"] {
            background-color: #253646 !important;
        }

        /* Заголовки внутри сайдбара */
        .sidebar-title {
            color: white;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
        }

        /* Текст в сайдбаре */
        .sidebar-text {
            color: white;
        }

        /* Скрываем верхнее меню, футер и хедер Streamlit */
        #MainMenu, footer, header {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)


# Текст в сайдбаре
st.sidebar.markdown('<div class="sidebar-title">Proiect: AI Recruiter pentru MAIB</div>', unsafe_allow_html=True)
# st.sidebar.markdown('<div class="sidebar-title">HR-RECRUITER</div>', unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.markdown("""
<div class="sidebar-text">

1. 📥 **Încărcarea posturilor vacante**

   *Agentul încarcă automat toate posturile vacante actuale de la MAIB.*

2. 📄 **CV-ul utilizatorului**

   *Utilizatorul își încarcă CV-ul pentru analiză.*

3. 🤖 **Căutarea posturilor potrivite**

   * *Agentul analizează CV-ul și identifică **top 3 posturi** relevante pentru experiența și competențele candidatului.*

4. 🔍 **Analiza celei mai relevante poziții**

   * *Evidențiază **punctele forte** ale candidatului.*
   * *Identifică **punctele slabe** sau lipsurile în competențe.*

5. ✅ **Acordul candidatului**

   *Dacă este interesat, candidatul își exprimă acordul pentru a continua procesul.*

6. 🗣️ **Primul interviu (general)**

   *Agentul pune întrebări generale, analizează răspunsurile și formulează **primele concluzii**.*

7. 💻 **Interviul tehnic**

   *Evaluarea competențelor tehnice ale candidatului în raport cu cerințele postului și furnizarea unui **feedback tehnic**.*

8. 📋 **Concluzia finală**

   *Agentul oferă un verdict final: **recomandare pentru angajare** sau **refuz argumentat**.*

---
</div>
""", unsafe_allow_html=True)


# Устанавливаем стиль для центрирования элементов
st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        /height: 5vh;
        text-align: center;
        flex-direction: column;
        margin-top: 0vh;  /* отступ сверху */
    }
    .github-icon:hover {
        color: #4078c0; /* Изменение цвета при наведении */
    }
    </style>
    <div class="center">
        <img src="https://www.maib.md/uploads/custom_blocks/image_1633004921_8nR1jw3Qfu_auto__0.png" width="300">
        <h1>HR-reqruiter</h1>        
    </div>
    """, unsafe_allow_html=True)

st.divider()
# Настройки для канвы
stroke_width = 10
stroke_color = "black"
bg_color = "white"
drawing_mode = "freedraw"

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

class DocumentChunk:
    def __init__(self, text, doc_name, page_num):
        self.text = text
        self.doc_name = doc_name
        self.page_num = page_num
        self.embedding = None

# class KnowledgeBase:
#     def __init__(self):
#         self.chunks = []
#         self.uploaded_files = []
#         self.vectorizer = TfidfVectorizer(stop_words='english')
#         self.tfidf_matrix = None
#         self.doc_texts = []
    
#     def split_text(self, text, max_tokens=2000):
#         paragraphs = text.split('\n\n')
#         chunks = []
#         current_chunk = ""
        
#         for para in paragraphs:
#             para = para.strip()
#             if not para:
#                 continue
                
#             tokens = tokenizer.tokenize(para)
#             if len(tokenizer.tokenize(current_chunk + para)) > max_tokens:
#                 if current_chunk:
#                     chunks.append(current_chunk)
#                     current_chunk = para
#                 else:
#                     chunks.append(para)
#                     current_chunk = ""
#             else:
#                 if current_chunk:
#                     current_chunk += "\n\n" + para
#                 else:
#                     current_chunk = para
        
#         if current_chunk:
#             chunks.append(current_chunk)
            
#         return chunks



    def load_text(self, text, file_name):
        if file_name in self.uploaded_files:
            return False

        chunks = self.split_text(text)
        for chunk in chunks:
            self.chunks.append(SimpleNamespace(text=chunk, source=file_name))
        self.uploaded_files.append(file_name)
        return True
    
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
                        chunks = self.split_text(page_text)
                        for chunk in chunks:
                            self.chunks.append(DocumentChunk(
                                text=chunk,
                                doc_name=file_name,
                                page_num=page_num + 1
                            ))
                            self.doc_texts.append(chunk)
                
                if self.chunks:
                    self.uploaded_files.append(file_name)
                    # Обновляем TF-IDF матрицу
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
                    return True
                else:
                    st.error(f"Не удалось извлечь текст из файла {file_name}")
                    return False
        except Exception as e:
            st.error(f"Ошибка загрузки PDF: {e}")
            return False
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def find_most_relevant_chunks(self, query, top_k=3):
        """Находит наиболее релевантные чанки с помощью TF-IDF и косинусного сходства"""
        if not self.chunks:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        return [(self.chunks[i].text, self.chunks[i].doc_name, self.chunks[i].page_num) 
                for i in top_indices if similarities[0][i] > 0.1]
    
    def get_document_names(self):
        return self.uploaded_files

# Инициализация
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()

if 'messages' not in st.session_state:
    st.session_state.messages = []

############################################
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json

#st.title("Парсер вакансий Moldova Agroindbank с rabota.md")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

base_url = "https://www.rabota.md/ru/companies/moldova-agroindbank#vacancies"

if st.button("Încarcă ofertele de muncă de pe rabota.md"):
    try:
        with st.spinner("Загружаем страницу вакансий..."):
            response = requests.get(base_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            links = soup.find_all('a', class_='vacancyShowPopup')
            urls = [urljoin(base_url, a['href']) for a in links]

        st.success(f"Найдено вакансий: {len(urls)}")

        vacancies_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, url in enumerate(urls):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                title_tag = soup.find('h1')
                title = title_tag.get_text(strip=True) if title_tag else 'Не найдено'

                vacancy_content = soup.find('div', class_='vacancy-content')
                description = vacancy_content.get_text(separator='\n', strip=True) if vacancy_content else 'Описание не найдено'

                vacancies_data.append({
                    'url': url,
                    'title': title,
                    'description': description
                })

                status_text.text(f"[{i+1}/{len(urls)}] Сохранена вакансия: {title}")
                progress_bar.progress((i+1)/len(urls))

                time.sleep(0.1)  # пауза для уважения сервера

            except Exception as e:
                st.error(f"Ошибка при обработке {url}: {e}")

        st.success("✅ Все вакансии загружены!")

        for vacancy in vacancies_data:
            st.markdown(f'<h4><a href="{vacancy["url"]}" style="color:#40c1ac; text-decoration:none;">{vacancy["title"]}</a></h3>', unsafe_allow_html=True)

        json_data = json.dumps(vacancies_data, ensure_ascii=False, indent=2)
        st.download_button("Скачать вакансии в JSON", data=json_data, file_name="vacancies.json", mime="application/json")

    except Exception as e:
        st.error(f"Ошибка при загрузке страницы вакансий: {e}")


#################################################################



# # Загрузка CV
# uploaded_files = st.file_uploader("Încarcă CV-ul tău în format PDF", type="pdf", accept_multiple_files=True)
# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         if uploaded_file.name not in st.session_state.knowledge_base.get_document_names():
#             success = st.session_state.knowledge_base.load_pdf(uploaded_file.getvalue(), uploaded_file.name)
#             if success:
#                 st.success(f"Fișierul {uploaded_file.name} a fost încărcat cu succes!")

# if not st.session_state.knowledge_base.get_document_names():
#     st.info("ℹ️ Încarcă CV-ul pentru a continua analiza.")
#     st.stop()

# # Анализ posturilor potrivite
# st.subheader("🔎 Identificăm posturile potrivite pentru tine...")

# # Получаем top-3 вакансии, релевантные содержимому CV
# cv_text = "\n\n".join([chunk.text for chunk in st.session_state.knowledge_base.chunks])
# top_k = 3
# vacancy_texts = [f"{v['title']} {v['description']}" for v in vacancies_data]
# vectorizer = TfidfVectorizer(stop_words='english')
# matrix = vectorizer.fit_transform([cv_text] + vacancy_texts)
# similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]
# top_indices = np.argsort(similarities)[-top_k:][::-1]

# # Показываем top-3 вакансии
# st.subheader("🏆 Top 3 posturi relevante")
# for i, idx in enumerate(top_indices):
#     vacancy = vacancies_data[idx]
#     st.markdown(f"### {i+1}. [{vacancy['title']}]({vacancy['url']})")
#     st.markdown(vacancy['description'])

# # Анализ самой релевантной вакансии
# best_vacancy = vacancies_data[top_indices[0]]
# context = f"CV-ul candidatului:\n{cv_text[:3000]}...\n\nPostul:\n{best_vacancy['title']} - {best_vacancy['description']}"

# prompt_analysis = f"""
# Evaluează compatibilitatea dintre CV-ul candidatului și acest post.
# - Identifică punctele forte (ce se potrivește bine).
# - Menționează ce competențe lipsesc sau sunt slabe.
# Răspunde în limba română.
# """.strip()

# data = {
#     "model": "deepseek-chat",
#     "messages": [
#         {"role": "user", "content": prompt_analysis},
#         {"role": "user", "content": context}
#     ],
#     "max_tokens": 1000,
#     "temperature": 0.2
# }

# st.subheader("🔍 Analiza celei mai relevante poziții")
# with st.spinner("Se generează analiza..."):
#     try:
#         response = requests.post(url, headers=headers, json=data)
#         if response.status_code == 200:
#             answer = response.json()['choices'][0]['message']['content']
#             st.markdown(answer + " ✅")
#         else:
#             st.error(f"❌ Eroare API: {response.status_code} - {response.text}")
#     except Exception as e:
#         st.error(f"❌ A apărut o eroare: {e}")
import streamlit as st
from pathlib import Path
import docx2txt
from PyPDF2 import PdfReader
import tempfile
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Класс для управления загруженными документами
class Chunk:
    def __init__(self, text, source):
        self.text = text
        self.source = source

class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.uploaded_files = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.doc_texts = []

    def split_text(self, text, max_tokens=2000):
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            tokens = tokenizer.tokenize(para)
            if len(tokenizer.tokenize(current_chunk + para)) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = para
                else:
                    chunks.append(para)
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _load_docx(self, file_content, file_name):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            text = docx2txt.process(tmp_file_path)
            self.chunks.append(Chunk(text=text, source=file_name))
            self.doc_names.add(file_name)
            return True
        except Exception as e:
            st.error(f"❌ Eroare la citirea DOCX: {e}")
            return False

    def _load_txt(self, file_content, file_name):
        try:
            text = file_content.decode("utf-8")
            self.chunks.append(Chunk(text=text, source=file_name))
            self.doc_names.add(file_name)
            return True
        except Exception as e:
            st.error(f"❌ Eroare la citirea TXT: {e}")
            return False

    def get_document_names(self):
        return list(self.doc_names)

# 🔹 Инициализация состояния
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()

# # 🔹 Данные вакансий (пример)
# vacancies_data = [
#     {"title": "Software Developer", "description": "We are looking for a Python developer with experience in ML.", "url": "https://example.com/dev"},
#     {"title": "Data Analyst", "description": "Candidate should know SQL, Excel and BI tools.", "url": "https://example.com/analyst"},
#     {"title": "DevOps Engineer", "description": "Looking for someone with AWS and CI/CD experience.", "url": "https://example.com/devops"},
#     {"title": "Frontend Developer", "description": "React.js knowledge is a must. Experience with Tailwind is a plus.", "url": "https://example.com/frontend"}
# ]

# 🔹 Интерфейс загрузки
# st.title("📄 Analizator CV & Potrivire Posturi")
uploaded_files = st.file_uploader(
    "Încarcă CV-ul tău (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

import docx2txt
import io

#uploaded_files = st.file_uploader("Încarcă CV-ul tău (PDF, DOCX sau TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name in st.session_state.knowledge_base.uploaded_files:
            continue

        file_bytes = uploaded_file.getvalue()

        if file_name.endswith(".pdf"):
            success = st.session_state.knowledge_base.load_pdf(file_bytes, file_name)

        elif file_name.endswith(".docx"):
            text = docx2txt.process(io.BytesIO(file_bytes))
            success = st.session_state.knowledge_base.load_text(text, file_name)

        elif file_name.endswith(".txt"):
            text = file_bytes.decode("utf-8")
            success = st.session_state.knowledge_base.load_text(text, file_name)

        else:
            st.warning(f"Formatul fișierului {file_name} nu este acceptat.")
            continue

        if success:
            st.success(f"Fișierul {file_name} a fost încărcat cu succes!")


if not st.session_state.knowledge_base.get_document_names():
    st.info("ℹ️ Încarcă CV-ul pentru a continua analiza.")
    st.stop()

# 🔹 Анализ
st.subheader("🔎 Identificăm posturile potrivite pentru tine...")

cv_text = "\n\n".join([chunk.text for chunk in st.session_state.knowledge_base.chunks])
top_k = 3
vacancy_texts = [f"{v['title']} {v['description']}" for v in vacancies_data]

vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform([cv_text] + vacancy_texts)
similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]
top_indices = np.argsort(similarities)[-top_k:][::-1]

# 🔹 Показываем топ вакансии
st.subheader("🏆 Top 3 posturi relevante")
for i, idx in enumerate(top_indices):
    vacancy = vacancies_data[idx]
    st.markdown(f"### {i+1}. [{vacancy['title']}]({vacancy['url']})")
    st.markdown(vacancy['description'])

# 🔹 Анализ лучшей вакансии
best_vacancy = vacancies_data[top_indices[0]]
context = f"CV-ul candidatului:\n{cv_text[:3000]}...\n\nPostul:\n{best_vacancy['title']} - {best_vacancy['description']}"

prompt_analysis = """
Evaluează compatibilitatea dintre CV-ul candidatului și acest post.
- Identifică punctele forte (ce se potrivește bine).
- Menționează ce competențe lipsesc sau sunt slabe.
Răspunde în limba română.
""".strip()

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": prompt_analysis},
        {"role": "user", "content": context}
    ],
    "max_tokens": 1000,
    "temperature": 0.2
}

# 🔹 Внешний API (замени своими ключами и URL)
url = "https://api.your-provider.com/v1/chat/completions"
headers = {"Authorization": f"Bearer YOUR_API_KEY"}

st.subheader("🔍 Analiza celei mai relevante poziții")
with st.spinner("Se generează analiza..."):
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            answer = response.json()['choices'][0]['message']['content']
            st.markdown(answer + " ✅")
        else:
            st.error(f"❌ Eroare API: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"❌ A apărut o eroare: {e}")

# Кнопка очистки чата
if st.button("Очистить чат"):
    st.session_state.messages = []
    st.rerun()
