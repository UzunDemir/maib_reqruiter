import os
import streamlit as st
import requests
import json
import time
from PyPDF2 import PdfReader
import tempfile
from datetime import datetime
#from dotenv import load_dotenv

# Загрузка переменных окружения
#load_dotenv()

# # Получение API ключа
# api_key = os.getenv("DEEPSEEK_API_KEY")
# if not api_key:
#     st.error("API ключ не найден. Пожалуйста, создайте файл .env с DEEPSEEK_API_KEY")
#     st.stop()

hide_github_icon = """
<style>
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK { 
    display: none !important; 
}
#MainMenu, footer, header { 
    display: none !important; 
}
</style>
"""

st.markdown(hide_github_icon, unsafe_allow_html=True)


st.sidebar.write("[Uzun Demir](https://uzundemir.github.io/)") #[Github](https://github.com/UzunDemir)     [Linkedin](https://www.linkedin.com/in/uzundemir/)     
st.sidebar.write("[Github](https://github.com/UzunDemir)")
st.sidebar.write("[Linkedin](https://www.linkedin.com/in/uzundemir/)")
st.sidebar.title("Описание проекта")
st.sidebar.title("TEST-passer (AI-ассистент по тестам)")
st.sidebar.divider()
st.sidebar.write(
        """
                                       
                     Это приложение выполнено в целях помощи студентам при сдаче тестов по ЛЮБОЙ образовательной теме.                  

                     1. Как это работает? 
                     
                        Студент загружает учебный материал в pdf. TEST-passer отвечает на тесты, выбирая правильные ответы. Точность ответов на тестировании составила 88%.
                     
                     2. Почему не воспользоваться обычными чатами (GPT, DeepSeek и т. д.)? 
                     
                        Несмотря на то что модель обучена на огромном 
                        количестве информации, она не понимает информацию, как человек, а лишь предсказывает "вероятный следующий фрагмент текста". 
                        Она также имеет способность "галлюцинировать", то есть может "придумать" факт, источник или термин, которого не существует, но звучит правдоподобно.
                        Поэтому наиболее правильные ответы будет выдавать модель, которая исплользует только НУЖНЫЙ иатериал. 
                        
                     3. Что делает приложение?    
                     
                        * Загружает и обрабатывает pdf-файлы (любые курсы, предметы, темы)
                        * Создает векторную базу данных
                        * Применяет динамический чанкинг (делит по смысловым границам)
                        * Гибридный поиск (HyDE + ключевые слова) 
                          (комбинирует два метода поиска, чтобы находить ответы, если он сформулирован иначе чем в учебных материалах)
                        * Валидация ответов
                        * Настройка DeepSeek для генерации ответов (можно использовать и другие модели) 
                     
                     4. Будут ли доработки?

                        Да, будут:

                        * возможность загрузки вопросов виде скриншотов
                        * комбинирование методов и моделей (ансамблирование) для получения максимально точных ответов
                        * уменьшение времени поиска ответа 
                     
                        
                     """
    )

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
        <img src="https://github.com/UzunDemir/mnist_777/blob/main/200w.gif?raw=true">
        <h1>TEST-passer</h1>
        <h2>AI-ассистент по тестам</h2>
        <p> (строго по учебным материалам)</p>
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

class KnowledgeBase:
    def __init__(self):
        self.documents = {}
        self.uploaded_files = []
    
    def load_pdf(self, file_content, file_name):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                self.documents[file_name] = text
                self.uploaded_files.append(file_name)
                return True
        except Exception as e:
            st.error(f"Ошибка загрузки PDF: {e}")
            return False
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def get_all_text(self):
        return "\n\n".join([f"### {name}\n{text}" for name, text in self.documents.items()])
    
    def get_document_names(self):
        return self.uploaded_files

# Инициализация
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Интерфейс Streamlit
#st.title("AI-ассистент по тестам (строго по учебным материалам)")

# Загрузка документов
uploaded_files = st.file_uploader("Загрузить PDF", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.knowledge_base.get_document_names():
            success = st.session_state.knowledge_base.load_pdf(uploaded_file.getvalue(), uploaded_file.name)
            if success:
                st.success(f"Файл {uploaded_file.name} успешно загружен")

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
    # Добавляем вопрос в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Подготавливаем контекст из базы знаний
    context = st.session_state.knowledge_base.get_all_text()
    full_prompt = f""""Answer strictly based on the educational materials provided below.
    Respond in the same language the question is written in.
    If the answer is not found in the materials, reply with: 'Answer not found in the materials'.
    
    Materials:
    {context}
    
    Question: {prompt}"""
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": True
    }
    
    # Показываем индикатор выполнения
    with st.spinner("Ищем ответ..."):
        start_time = datetime.now()
        
        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            
            if response.status_code == 200:
                full_response = ""
                message_placeholder = st.empty()
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        if decoded_line.startswith("data:"):
                            json_data = decoded_line[5:].strip()
                            
                            if json_data == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(json_data)
                                if 'choices' in data and len(data['choices']) > 0:
                                    chunk_content = data['choices'][0]['delta'].get('content', '')
                                    if chunk_content:
                                        full_response += chunk_content
                                        message_placeholder.markdown(f"🤖 {full_response}")
                                        time.sleep(0.05)
                            except json.JSONDecodeError:
                                continue
                
                # Добавляем ответ в историю
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                with st.chat_message("assistant"):
                    if "ответ не найден" in full_response.lower():
                        st.markdown(f"{full_response} ❌")
                    else:
                        st.markdown(f"{full_response} ✅")
                
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
