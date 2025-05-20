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
import time
import json

st.title("Парсер вакансий MAIB")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

# Ввод ссылок вакансий или загрузка из текстового поля (для примера)
urls_text = st.text_area("Вставьте ссылки вакансий (по одной в строке):")

if st.button("Загрузить вакансии"):
    if not urls_text.strip():
        st.warning("Пожалуйста, введите хотя бы одну ссылку.")
    else:
        urls = urls_text.strip().split('\n')
        vacancies_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, url in enumerate(urls):
            try:
                response = requests.get(url.strip(), headers=headers)
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

                progress_bar.progress((i + 1) / len(urls))
                time.sleep(1)  # пауза для уважения сервера

            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка сети при обработке {url}: {e}")
            except Exception as e:
                st.error(f"Ошибка при обработке {url}: {e}")

        st.success(f"Загрузка вакансий завершена! Найдено вакансий: {len(vacancies_data)}")

        # Вывод результатов
        for vacancy in vacancies_data:
            st.subheader(vacancy['title'])
            st.write(vacancy['description'])
            st.markdown(f"[Ссылка на вакансию]({vacancy['url']})")

        # Возможность скачать результаты
        json_data = json.dumps(vacancies_data, ensure_ascii=False, indent=4)
        st.download_button("Скачать вакансии в JSON", data=json_data, file_name="vacancies.json", mime="application/json")

#################################################################



# Загрузка документов
uploaded_files = st.file_uploader("Încarcă CV-ul tău în format PDF", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.knowledge_base.get_document_names():
            success = st.session_state.knowledge_base.load_pdf(uploaded_file.getvalue(), uploaded_file.name)
            if success:
                st.success(f"Fișierul {uploaded_file.name} a fost încărcat cu succes!")

# Отображение загруженных документов
if st.session_state.knowledge_base.get_document_names():
    st.subheader("📚 Загруженные документы:")
    for doc in st.session_state.knowledge_base.get_document_names():
        st.markdown(f"- {doc}")
else:
    st.info("ℹ️ Документы не загружены")

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ввод вопроса
if prompt := st.chat_input("Введите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Поиск наиболее релевантных чанков
    relevant_chunks = st.session_state.knowledge_base.find_most_relevant_chunks(prompt)
    
    if not relevant_chunks:
        response_text = "Ответ не найден в материалах ❌"
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
    else:
        # Формируем контекст из релевантных чанков
        context = "\n\n".join([f"Документ: {doc_name}, страница {page_num}\n{text}" 
                             for text, doc_name, page_num in relevant_chunks])
        
        full_prompt = f"""Answer strictly based on the educational materials provided below.
     Respond in the same language the question is written in.
     If the answer is not found in the materials, reply with: 'Answer not found in the materials'.
    
    
        
        educational materials: {prompt}
        
        relevant materials:
        {context}"""
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 2000,
            "temperature": 0.1  # Уменьшаем случайность ответов
        }
        
        with st.spinner("Ищем ответ..."):
            start_time = datetime.now()
            
            try:
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    full_response = response_data['choices'][0]['message']['content']
                    
                    # Добавляем ссылки на источники
                    sources = "\n\nИсточники:\n" + "\n".join(
                        [f"- {doc_name}, стр. {page_num}" for _, doc_name, page_num in relevant_chunks]
                    )
                    full_response += sources
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    with st.chat_message("assistant"):
                        st.markdown(full_response + " ✅")
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    st.info(f"⏱️ Поиск ответа занял {duration:.2f} секунд")
                else:
                    st.error(f"Ошибка API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")

# Кнопка очистки чата
if st.button("Очистить чат"):
    st.session_state.messages = []
    st.rerun()
