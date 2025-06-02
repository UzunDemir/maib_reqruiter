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
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- Language Settings ---
if 'language' not in st.session_state:
    st.session_state.language = 'rom'  # Default language

def get_translation(key):
    translations = {
        'rom': {
            'app_title': 'AI HR-Recruiter',
            'sidebar_title': 'MAIB AI HR-Recruiter',
            'load_vacancies': 'Încarcă ofertele de muncă pentru tine...',
            'vacancies_list': 'Lista ofertelor MAIB:',
            'upload_cv': 'Încărcă CV-ul tău (PDF, DOCX, TXT)',
            'best_matches': 'Cele mai relevante oferte pentru CV-ul tău',
            'generate_analysis': 'Generează analiza de potrivire',
            'detailed_analysis': 'Analiză detaliată a conformității',
            'download_analysis': 'Descarcă analiza (DOCX)',
            'start_interview': 'A trece interviul introductiv',
            'interview_in_progress': 'Interviul a început! Vă rog să răspundeți la întrebările de mai jos.',
            'finish_interview': 'Interviul s-a încheiat',
            'start_technical': 'Începe interviul tehnic',
            'technical_in_progress': 'Interviul tehnic a început! Vă rugăm să răspundeți la întrebările de mai jos.',
            'finish_technical': 'Finalizează interviul tehnic',
            'technical_feedback': 'Feedback tehnic',
            'final_conclusion': 'Concluzia finală',
            'reset_process': 'Resetează procesul',
            'candidate_profile': 'Profilul candidatului',
            'download_profile': 'Descarcă profilul candidatului (DOCX)',
            'current_step': 'Etapa curentă:',
            'step1': 'Încărcare CV',
            'step2': 'Analiză potrivire',
            'step3': 'Interviu general',
            'step4': 'Interviu tehnic',
            'step5': 'Concluzie finală'
        },
        'rus': {
            'app_title': 'AI HR-Рекрутер',
            'sidebar_title': 'MAIB AI HR-Рекрутер',
            'load_vacancies': 'Загрузить вакансии для вас...',
            'vacancies_list': 'Список вакансий MAIB:',
            'upload_cv': 'Загрузите ваше резюме (PDF, DOCX, TXT)',
            'best_matches': 'Самые подходящие вакансии для вашего резюме',
            'generate_analysis': 'Сгенерировать анализ соответствия',
            'detailed_analysis': 'Подробный анализ соответствия',
            'download_analysis': 'Скачать анализ (DOCX)',
            'start_interview': 'Начать общее собеседование',
            'interview_in_progress': 'Собеседование начато! Пожалуйста, ответьте на вопросы ниже.',
            'finish_interview': 'Завершить собеседование',
            'start_technical': 'Начать техническое собеседование',
            'technical_in_progress': 'Техническое собеседование начато! Пожалуйста, ответьте на вопросы ниже.',
            'finish_technical': 'Завершить техническое собеседование',
            'technical_feedback': 'Технический отзыв',
            'final_conclusion': 'Заключительный вывод',
            'reset_process': 'Сбросить процесс',
            'candidate_profile': 'Профиль кандидата',
            'download_profile': 'Скачать профиль кандидата (DOCX)',
            'current_step': 'Текущий этап:',
            'step1': 'Загрузка резюме',
            'step2': 'Анализ соответствия',
            'step3': 'Общее собеседование',
            'step4': 'Техническое собеседование',
            'step5': 'Заключительный вывод'
        },
        'en': {
            'app_title': 'AI HR-Recruiter',
            'sidebar_title': 'MAIB AI HR-Recruiter',
            'load_vacancies': 'Load job vacancies for you...',
            'vacancies_list': 'List of MAIB vacancies:',
            'upload_cv': 'Upload your CV (PDF, DOCX, TXT)',
            'best_matches': 'Most relevant job matches for your CV',
            'generate_analysis': 'Generate matching analysis',
            'detailed_analysis': 'Detailed compatibility analysis',
            'download_analysis': 'Download analysis (DOCX)',
            'start_interview': 'Start introductory interview',
            'interview_in_progress': 'Interview has started! Please answer the questions below.',
            'finish_interview': 'Interview completed',
            'start_technical': 'Start technical interview',
            'technical_in_progress': 'Technical interview has started! Please answer the questions below.',
            'finish_technical': 'Complete technical interview',
            'technical_feedback': 'Technical feedback',
            'final_conclusion': 'Final conclusion',
            'reset_process': 'Reset process',
            'candidate_profile': 'Candidate profile',
            'download_profile': 'Download candidate profile (DOCX)',
            'current_step': 'Current step:',
            'step1': 'CV Upload',
            'step2': 'Matching Analysis',
            'step3': 'General Interview',
            'step4': 'Technical Interview',
            'step5': 'Final Conclusion'
        }
    }
    return translations[st.session_state.language].get(key, key)

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
        .current-step {
            background-color: #40c1ac;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
    </style>
    <div class="center">
        <img src="https://www.maib.md/uploads/custom_blocks/image_1633004921_8nR1jw3Qfu_auto__0.png" width="300">
        <h1>{}</h1>
    </div>
""".format(get_translation('app_title')), unsafe_allow_html=True)
