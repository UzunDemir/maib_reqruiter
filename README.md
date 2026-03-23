# 🤖 AI HR-Recruiter

**Intelligent recruitment system with automated CV analysis, interviews, and candidate evaluation**

---

## ✨ Features

| Module | Description |
|--------|-------------|
| **📥 Vacancy Loading** | Automatic scraping of job vacancies from MAIB career portal with parallel processing |
| **📄 CV Analysis** | Upload CV (PDF/DOCX/TXT) — TF-IDF based matching with job descriptions |
| **🎯 Smart Matching** | Top-3 job recommendations with similarity scores and visual progress bars |
| **📊 Detailed Analysis** | AI-powered compatibility report: strengths, gaps, recommendations, match percentage |
| **🗣️ General Interview** | Dynamic question generation based on CV + AI response analysis (human vs AI detection) |
| **💻 Technical Interview** | Role-specific technical questions with automated scoring and feedback |
| **📑 Candidate Profile** | Structured profile generation: skills, experience, motivation, salary expectations |
| **✅ Final Decision** | Automated recommendation: hire / hire with reserves / reject with reasoning |
| **🌍 Multi-language** | UI in **Română / Русский / English** |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                          │
├─────────────────────────────────────────────────────────────────┤
│  Sidebar (Lang / Steps / Vacancies)  │  Main Panel             │
├─────────────────────────────────────────────────────────────────┤
│                    ORCHESTRATION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Vacancy  │ │   CV     │ │ Matching │ │Interview │          │
│  │ Loader   │ │ Processor│ │ Engine   │ │ Manager  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                      DEEPSEEK LLM API                          │
│  • Question generation  • Analysis  • Feedback  • Detection    │
├─────────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  PDF/    │ │   TF-IDF │ │ Session  │ │   DOCX   │          │
│  │  DOCX    │ │  Vector  │ │  State   │ │  Export  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/your-repo/ai-hr-recruiter.git
cd ai-hr-recruiter
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.streamlit/secrets.toml`:
```toml
DEEPSEEK_API_KEY = "your-api-key-here"
```

### 3. Run Application
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
ai-hr-recruiter/
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── secrets.toml          # API keys (not in repo)
└── README.md
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | DeepSeek API |
| Document Parsing | PyPDF2, python-docx |
| Text Processing | scikit-learn (TF-IDF, cosine similarity) |
| Web Scraping | BeautifulSoup, requests |
| Parallel Processing | ThreadPoolExecutor |
| Export | python-docx |

---

## 🧠 Agent Workflow

```
1. Load Vacancies → 2. Upload CV → 3. Match Analysis → 
4. General Interview → 5. Technical Interview → 6. Final Recommendation
```

Each step is saved in `st.session_state` — process can be resumed anytime.

---

## 📊 Sample Outputs

- **Match Analysis**: Strength/weakness breakdown + % match
- **Candidate Profile**: Professional portrait, motivation, salary expectations
- **Technical Feedback**: Detailed evaluation + score (0–10)
- **Final Report**: Complete DOCX with all stages

---

## ⚠️ Important Notes

- Vacancy source: `rabota.md` (MAIB company page) — URL may need updating
- LLM responses may vary; human oversight recommended for final decisions
- AI-generated answer detection is experimental

---

## 🔮 Future Improvements

- [ ] Persistent database for candidates
- [ ] Email notifications
- [ ] Video interview integration
- [ ] More sophisticated anti-cheating
- [ ] Docker deployment

---

## 👨‍💻 Author

**Uzun Demir**

---

## 📄 License

MIT
