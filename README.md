# Insighto: Candidate Analyzer

Insighto is an intelligent candidate analysis tool that leverages both resumes and LinkedIn profiles to generate comprehensive, recruiter-friendly reports. The app is built with Streamlit for an interactive web experience and integrates advanced NLP, machine learning, and LLMs for deep candidate insights.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Set your Apify API token in your environment for LinkedIn extraction.
3. Start the app:
   ```bash
   streamlit run streamlit_app.py
   ```
4. Open the provided local URL in your browser.

## How It Works

- **Resume Parsing:**  
  Insighto uses two parsing engines:
  - A regex/pattern-based parser for robust fallback extraction.
  - An intelligent parser powered by NLP, ML, and OCR for domain-agnostic section and entity detection.
  Both approaches extract key sections (education, experience, skills, projects, certifications, publications) and basic info.

- **LinkedIn Profile Extraction:**  
  The app fetches and structures LinkedIn profile data using the Apify API, extracting headline, experience, education, skills, projects, and recommendations.

- **Comprehensive Analysis:**  
  Parsed resume and LinkedIn data are processed and merged. If available, an LLM (Ollama) generates a professional, actionable report for recruiters. If the LLM is unavailable, a fallback summary is provided.

- **User Experience:**  
  Users can upload resumes (PDF, DOCX, TXT) and/or enter a LinkedIn URL. The app displays quick stats, allows report downloads, and provides debug info for transparency.

## Technical Insights

- **NLP & ML:**  
  Utilizes spaCy, transformers, NLTK, and sentence-transformers for section classification, entity extraction, and semantic analysis.
- **OCR:**  
  Integrates EasyOCR and Tesseract for extracting text from image-based resumes.
- **Streamlit UI:**  
  Offers a clean, two-column interface for resume upload and LinkedIn input, with real-time feedback and report generation.
- **Extensibility:**  
  Modular design allows easy integration of new parsers, LLMs, or data sources.

## Project Structure

- `streamlit_app.py`: Main app interface.
- `profile_extractor.py`: LinkedIn data extraction.
- `profile_processor.py`: Data processing and report generation.
- `resume_parser.py` & `intelligent_resume_parser.py`: Resume parsing engines.
- `requirements.txt`: Python dependencies.

---
