import streamlit as st
from PyPDF2 import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="MAIB CV Matcher", layout="wide")
st.title("MAIB CV Matcher")

# Функция для чтения текста из PDF
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Функция для чтения текста из DOCX
def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Функция для чтения всех вакансий
def read_job_descriptions(folder_path="job_descriptions"):
    jobs = []
    filenames = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            with open(path, "rb") as f:
                jobs.append(read_pdf(f))
                filenames.append(filename)
        elif filename.endswith(".docx"):
            jobs.append(read_docx(path))
            filenames.append(filename)
    return jobs, filenames

# Функция сопоставления
def find_most_relevant_jobs(cv_text, job_texts, job_filenames, top_n=3):
    texts = [cv_text] + job_texts
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(job_filenames[i], similarities[i]) for i in top_indices]

# Интерфейс загрузки резюме
uploaded_file = st.file_uploader("Загрузите своё резюме (PDF или DOCX)", type=["pdf", "docx"])

if uploaded_file:
    # Определяем тип и извлекаем текст
    if uploaded_file.name.endswith(".pdf"):
        cv_text = read_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        cv_text = read_docx(uploaded_file)

    st.subheader("🧾 Извлечённый текст резюме:")
    st.write(cv_text[:1000] + "..." if len(cv_text) > 1000 else cv_text)

    # Загружаем вакансии и ищем релевантные
    job_texts, job_filenames = read_job_descriptions()
    relevant_jobs = find_most_relevant_jobs(cv_text, job_texts, job_filenames)

    st.subheader("🔎 Топ релевантных вакансий:")
    for job_file, score in relevant_jobs:
        st.markdown(f"- **{job_file}** — релевантность: `{score:.2f}`")

