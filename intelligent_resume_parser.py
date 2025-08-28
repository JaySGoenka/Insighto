import re
import json
import os
import logging
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import spacy
import torch
import nltk
import chardet

from transformers import pipeline
from sentence_transformers import SentenceTransformer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import textstat
from textblob import TextBlob

import cv2
import fitz  
import pytesseract
import easyocr

from docx import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


# Declaring a dataclass to represent parsed resume sections
@dataclass
class ParsedSection:
    """Parsed resume sections"""
    name: str
    content: str
    confidence: float
    entities: List[Dict[str, Any]]
    start_line: int
    end_line: int


# Declaring a dataclass to represent extracted resume entities - which include personal information, education, work experience, skills, and certifications
# NER parser is used to detect named entities in the resume and then classify them into the appropriate sections 
@dataclass
class ResumeEntity:
    """Extracted entities from resume"""
    text: str
    label: str
    confidence: float
    start_char: int
    end_char: int
    metadata: Dict[str, str]


class SectionClassifier:
    """Section classifier using semantic similarity."""
    
    def __init__(self):
        
        self.universal_sections = {
            'education': ['education', 'academic', 'academics', 'qualifications', 'degrees', 'academic background'],
            'skills': ['skills', 'competencies', 'expertise', 'capabilities', 'proficiencies', 'abilities'],
            'experience': ['experience', 'work experience', 'employment', 'work history', 'professional experience', 'career history'],
            'projects': ['projects', 'project experience', 'portfolio', 'achievements', 'key projects', 'selected projects'],
            'certifications': ['certifications', 'certificates', 'licenses', 'accreditations', 'professional credentials'],
            'publications': ['publications', 'research', 'papers', 'articles', 'journals', 'conferences']
        }
        
        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_available = True
    
    
    def classify_section(self, text: str, context: str = "") -> Tuple[str, float]:
        """Classify text as a resume section using semantic similarity."""
        
        if not text.strip():
            return "unknown", 0.0
        
        text_lower = text.lower().strip()
        best_match = "unknown"
        best_score = 0.0
        
        if self.semantic_available:

            text_embedding = self.sentence_model.encode([text])
            
            for section_name, keywords in self.universal_sections.items():
                
                section_text = " ".join(keywords)
                section_embedding = self.sentence_model.encode([section_text])
                
                similarity = cosine_similarity(text_embedding, section_embedding)[0][0]
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = section_name
        
        # Boost confidence for exact or close matches
        for section_name, keywords in self.universal_sections.items():
            for keyword in keywords:
                if keyword in text_lower or any(keyword in text_lower for keyword in keywords):
                    best_score = min(1.0, best_score + 0.2)
                    break
        
        return best_match, min(1.0, best_score)


class EntityExtractor:
    """Domain-agnostic entity extraction using NLP."""
    
    def __init__(self):
        try:
            self.nlp_sm = spacy.load("en_core_web_sm")
            self.nlp_lg = spacy.load("en_core_web_lg")
            self.nlp_trf = spacy.load("en_core_web_trf")
            self.spacy_available = True
        except OSError:
            try:
                self.nlp_sm = spacy.load("en_core_web_sm")
                self.nlp_lg = None
                self.nlp_trf = None
                self.spacy_available = True
            except:
                self.spacy_available = False
                logger.warning("Spacy models not available")
        
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        self.transformers_available = True
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    # Entities picked by NER from a resume typically include names, organizations, locations, dates, and skills.
    def extract_entities(self, text: str, section_type: str = None) -> List[ResumeEntity]:
        """Extract entities using NLP"""
        
        entities = []
        
        if self.spacy_available:
            
            doc = self.nlp_sm(text)

            for ent in doc.ents:
                entities.append(ResumeEntity(
                    text=ent.text,
                    label=ent.label_,
                    confidence=0.8,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    metadata={'source': 'spacy', 'section': section_type}
                ))
        
        # NER (Named Entity Recognition) is used to detect entities from the resume
        if self.transformers_available:
            
            try:
                ner_results = self.ner_pipeline(text)
                
                for result in ner_results:
                    entities.append(ResumeEntity(
                        text=result['word'],
                        label=result['entity'],
                        confidence=result['score'],
                        start_char=0,  
                        end_char=len(result['word']),
                        metadata={'source': 'transformers', 'section': section_type}
                    ))
            except Exception as e:
                logger.warning(f"Transformers NER failed: {e}")
        
        entities = self.deduplicate_entities(entities)
        
        return entities
    

    def deduplicate_entities(self, entities: List[ResumeEntity]) -> List[ResumeEntity]:
        """Remove duplicate entities and merge similar ones."""
        
        if not entities:
            return entities
        
        entity_dict = {}
        for entity in entities:
            normalized_text = entity.text.lower().strip()
            if normalized_text not in entity_dict or entity.confidence > entity_dict[normalized_text].confidence:
                entity_dict[normalized_text] = entity
            else:
                entity_dict[normalized_text].metadata.update(entity.metadata)
        return list(entity_dict.values())


class IntelligentResumeParser:
    """resume parser using intelligent section detection."""
    
    def __init__(self, use_ocr: bool = True, use_ml: bool = True):
        self.use_ocr = use_ocr
        self.use_ml = use_ml
        
        # Initialize components
        self.section_classifier = SectionClassifier()
        self.entity_extractor = EntityExtractor()
        
        # Initialize OCR readers
        if self.use_ocr:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                self.paddle_ocr = None
            except:
                self.easyocr_reader = None
                logger.warning("EasyOCR not available")
        
        # Text processing
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        logger.info(f"Domain-Agnostic Resume Parser initialized - OCR: {self.use_ocr}, ML: {self.use_ml}")
    

    def ocr_image(self, image: np.ndarray) -> str:
        """Perform OCR on image using multiple engines."""
        
        text = ""
        
        # Try EasyOCR first
        if self.easyocr_reader:
            try:
                results = self.easyocr_reader.readtext(image)
                text = " ".join([result[1] for result in results])
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Fallback to Tesseract
        if not text:
            try:
                text = pytesseract.image_to_string(image)
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")
        
        return text

    
    def read_pdf(self, file_path: str) -> Optional[str]:
        """Read PDF using multiple strategies."""
        
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            
            # If text extraction is poor, use OCR
            if len(page_text.strip()) < 100 and self.use_ocr:
                logger.info(f"Poor text extraction on page {page_num}, using OCR")
                pix = page.get_pixmap()
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                ocr_text = self._ocr_image(img_array)
                text += ocr_text + "\n"
            else:
                text += page_text + "\n"
        
        doc.close()
        return text


    def read_image_with_ocr(self, file_path: str) -> Optional[str]:
        """Read image file using OCR."""
        
        try:
            img = cv2.imread(file_path)
            if img is None:
                return None
            
            return self._ocr_image(img)
        except Exception as e:
            logger.error(f"Image reading failed: {e}")
            return None


    def read_text(self, file_path: str) -> Optional[str]:
        """Read text file with encoding detection."""
        
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] or 'utf-8'
            
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Text file reading failed: {e}")
            return None
    
    def read_docx(self, file_path: str) -> Optional[str]:
        """Read DOCX file."""
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"DOCX reading failed: {e}")
            return None


    def read_file_intelligently(self, file_path: str) -> Optional[str]:
        """Read file content using OCR when needed."""
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.txt':
            return self.read_text(file_path)
        elif file_extension == '.pdf':
            return self.read_pdf(file_path)
        elif file_extension == '.docx':
            return self.read_docx(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return self.read_image_with_ocr(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return None
    

    def _extract_basic_info_nlp(self, content: str) -> Dict[str, Any]:
        """Extract basic information using NLP techniques."""
        
        basic_info = {
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "summary": "",
            "confidence": {}
        }
        
        # Extract email using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, content)
        if email_match:
            basic_info["email"] = email_match.group()
            basic_info["confidence"]["email"] = 0.95
        
        # Extract phone using multiple patterns
        phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US format
            r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b'  # International
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, content)
            if phone_match:
                basic_info["phone"] = phone_match.group()
                basic_info["confidence"]["phone"] = 0.9
                break
        
        # Extract name using NLP
        if self.entity_extractor.spacy_available:
            doc = self.entity_extractor.nlp_sm(content[:2000])  # Check first 2000 chars
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.split()) <= 4:
                    basic_info["name"] = ent.text
                    basic_info["confidence"]["name"] = 0.85
                    break
        
        # Extract location using NLP
        if self.entity_extractor.spacy_available:
            doc = self.entity_extractor.nlp_sm(content)
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:
                    basic_info["location"] = ent.text
                    basic_info["confidence"]["location"] = 0.8
                    break
        
        return basic_info


    def segment_content(self, content: str) -> Dict[str, str]:
        """segment content into sections using ML and NLP."""
        
        lines = content.split('\n')
        sections = {}
        
        # Use sliding window approach for better context
        window_size = 5
        current_section = "unknown"
        current_content = []
        
        for i in range(len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            # Create context window
            context_start = max(0, i - window_size // 2)
            context_end = min(len(lines), i + window_size // 2 + 1)
            context = " ".join(lines[context_start:context_end])
            
            # Classify this line/context
            section_name, confidence = self.section_classifier.classify_section(line, context)
            
            # If confidence is high enough, start new section
            if confidence > 0.6 and section_name != "unknown":
                if current_content and current_section != "unknown":
                    sections[current_section] = "\n".join(current_content)
                
                current_section = section_name
                current_content = [line]
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content and current_section != "unknown":
            sections[current_section] = "\n".join(current_content)
        
        return sections
    

    def parse_section(self, section_name: str, content: str) -> Dict[str, Any]:
        """Parse section content using intelligent NLP techniques."""
        
        if not content.strip():
            return {"content": "", "entities": [], "confidence": 0.0}
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(content, section_name)
        
        # Parse based on section type 
        parsed_data = {"content": content, "entities": []}
        
        if section_name == "experience":
            parsed_data.update(self.parse_experience(content, entities))
        elif section_name == "education":
            parsed_data.update(self.parse_education(content, entities))
        elif section_name == "skills":
            parsed_data.update(self.parse_skills(content, entities))
        elif section_name == "projects":
            parsed_data.update(self.parse_projects(content, entities))
        elif section_name == "certifications":
            parsed_data.update(self.parse_certifications(content, entities))
        elif section_name == "publications":
            parsed_data.update(self.parse_publications(content, entities))
        else:
            # Generic parsing for other sections
            parsed_data["confidence"] = 0.7
            parsed_data["entities"] = [entity.__dict__ for entity in entities]
        
        return parsed_data
    
    def parse_experience(self, content: str, entities: List[ResumeEntity]) -> Dict[str, Any]:
        """Parse experience section without domain assumptions."""
        
        experiences = []
        
        # Split into potential job entries using intelligent line analysis
        lines = content.split('\n')
        current_job = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for job titles and companies in entities
            for entity in entities:
                if entity.label in ['JOB_TITLE', 'COMPANY', 'ORG']:
                    if entity.label in ['JOB_TITLE']:
                        if current_job:
                            experiences.append(current_job)
                        current_job = {"title": entity.text, "company": "", "duration": "", "description": []}
                    elif entity.label in ['COMPANY', 'ORG'] and current_job:
                        current_job["company"] = entity.text
            
            # Add description lines
            if current_job and line and not line.startswith(('•', '-', '*', '#')):
                current_job["description"].append(line)
        
        if current_job:
            experiences.append(current_job)
        
        return {
            "experiences": experiences,
            "confidence": 0.8,
            "entities": [entity.__dict__ for entity in entities]
        }
    
    def parse_education(self,content: str, entities: List[ResumeEntity]) -> Dict[str, Any]:
        """Parse education section without domain assumptions."""
        education = []
        
        # Split into potential education entries
        lines = content.split('\n')
        current_edu = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for degree patterns in entities
            for entity in entities:
                if entity.label in ['DEGREE', 'ORG', 'DATE']:
                    if entity.label == 'DEGREE':
                        if current_edu:
                            education.append(current_edu)
                        current_edu = {"degree": entity.text, "institution": "", "duration": "", "field": ""}
                    elif entity.label == 'ORG' and current_edu:
                        current_edu["institution"] = entity.text
                    elif entity.label == 'DATE' and current_edu:
                        current_edu["duration"] = entity.text
            
            # Add description lines
            if current_edu and line and not line.startswith(('•', '-', '*', '#')):
                current_edu["description"].append(line)
        
        if current_edu:
            education.append(current_edu)
        
        return {
            "education": education,
            "confidence": 0.8,
            "entities": [entity.__dict__ for entity in entities]
        }
    
    def parse_skills(self, content: str, entities: List[ResumeEntity]) -> Dict[str, Any]:
        """Parse skills section without domain assumptions."""
        
        skills = []
        
        # Extract skills from entities
        for entity in entities:
            if entity.label in ['SKILL', 'TECHNOLOGY', 'COMPETENCY']:
                skills.append(entity.text)
        
        # Extract additional skills using text analysis
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split by common delimiters
            skill_parts = re.split(r'[,•\-\*\|]', line)
            for skill in skill_parts:
                skill = skill.strip()
                if skill and len(skill) > 1 and len(skill) < 50:
                    if not skill.isupper() and len(skill.split()) <= 3:
                        skills.append(skill)
        
        return {
            "skills": list(set(skills)),
            "confidence": 0.85,
            "entities": [entity.__dict__ for entity in entities]
        }
    
    def parse_projects(self, content: str, entities: List[ResumeEntity]) -> Dict[str, Any]:
        """Parse projects section without domain assumptions."""
        
        projects = []
        
        # Split into potential project entries
        lines = content.split('\n')
        current_project = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for project titles
            if re.search(r'^[A-Z][^:]+(?:[:]|$)', line) and len(line) < 100:
                if current_project:
                    projects.append(current_project)
                current_project = {"title": line.replace(':', '').strip(), "description": [], "technologies": []}
            
            # Add description
            elif current_project and line:
                current_project["description"].append(line)
        
        if current_project:
            projects.append(current_project)
        
        return {
            "projects": projects,
            "confidence": 0.8,
            "entities": [entity.__dict__ for entity in entities]
        }
    
    def parse_certifications(self, content: str, entities: List[ResumeEntity]) -> Dict[str, Any]:
        """Parse certifications section without domain assumptions."""
        certifications = []
        
        # Split into potential certification entries
        lines = content.split('\n')
        current_cert = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for certification names
            if re.search(r'^[A-Z][^:]+(?:[:]|$)', line) and len(line) < 100:
                if current_cert:
                    certifications.append(current_cert)
                
                cert_name = line.replace(':', '').strip()
                current_cert = {"name": cert_name, "issuer": "", "date": ""}
            
            # Add description
            elif current_cert and line:
                current_cert["description"].append(line)
        
        if current_cert:
            certifications.append(current_cert)
        
        return {
            "certifications": certifications,
            "confidence": 0.8,
            "entities": [entity.__dict__ for entity in entities]
        }
    
    def parse_publications(self, content: str, entities: List[ResumeEntity]) -> Dict[str, Any]:
        """Parse publications section without domain assumptions."""
        publications = []
        
        # Split into potential publication entries
        lines = content.split('\n')
        current_pub = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for publication titles
            if re.search(r'^[A-Z][^:]+(?:[:]|$)', line) and len(line) < 150:
                if current_pub:
                    publications.append(current_pub)
                
                pub_title = line.replace(':', '').strip()
                current_pub = {"title": pub_title, "authors": "", "journal": "", "year": ""}
            
            # Add description
            elif current_pub and line:
                current_pub["description"].append(line)
        
        if current_pub:
            publications.append(current_pub)
        
        return {
            "publications": publications,
            "confidence": 0.8,
            "entities": [entity.__dict__ for entity in entities]
        }
    
    def calculate_overall_confidence(self, parsed_sections: Dict[str, Any]) -> float:
        """Calculate overall parsing confidence."""
        if not parsed_sections:
            return 0.0
        
        total_confidence = 0.0
        section_count = 0
        
        for section_name, section_data in parsed_sections.items():
            if isinstance(section_data, dict) and "confidence" in section_data:
                total_confidence += section_data["confidence"]
                section_count += 1
        
        return total_confidence / section_count if section_count > 0 else 0.0
    

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Main method to parse resume"""
        
        try:
            content = self.read_file_intelligently(file_path)
            if not content:
                return {"error": "Failed to read file"}
            
            # Extract basic information using NLP
            basic_info = self._extract_basic_info_nlp(content)
            
            # Segment content intelligently
            sections = self.segment_content(content)

            parsed_sections = {}
            for section_name, section_content in sections.items():
                parsed_sections[section_name] = self.parse_section(section_name, section_content)

            # Generate comprehensive output
            parsed_resume = {
                "basic_info": basic_info,
                "sections": parsed_sections,
                "metadata": {
                    "parsing_method": "ML/NLP",
                    "ocr_used": self.use_ocr,
                    "ml_used": self.use_ml,
                    "confidence_score": self.calculate_overall_confidence(parsed_sections),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            return parsed_resume
            
        except Exception as e:
            logger.error(f"Parsing failed: {str(e)}")
            return {"error": f"Parsing failed: {str(e)}"}


def main():
    """Test the intelligent resume parser."""
    parser = IntelligentResumeParser(use_ocr=True, use_ml=True)

    print("Testing Intelligent Resume Parser...")
    resume_path = "/Users/jay/Desktop/JaySGoenka Resume.pdf"
    print(f"Parsing resume: {resume_path}")
    parsed_data = parser.parse_resume(resume_path)
    if "error" not in parsed_data:
        print("Resume parsed successfully!")
        print("\nParsed Sections Only:")

        def convert_numpy(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # Only keep relevant content for specified sections
        section_keys = ["education", "skills", "experience", "projects", "certifications", "publications"]
        filtered_sections = {}
        for key in section_keys:
            section = parsed_data.get("sections", {}).get(key)
            if section and "content" in section:
                filtered_sections[key] = section["content"]
        
        # Include basic info and sections for LLM
        output_data = {
            "basic_info": parsed_data.get("basic_info", {}),
            "sections": filtered_sections
        }
        
        print(json.dumps(output_data, indent=2, ensure_ascii=False, default=convert_numpy))
        with open("parsed_resume.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=convert_numpy)
            print("\nFiltered sections and basic info saved to 'parsed_resume.json'")

if __name__ == "__main__":
    main()
