import re
import json
import spacy
from typing import Dict, List, Any, Optional, Tuple
import os
import PyPDF2
from docx import Document


class ResumeParser:
    """Resume parser using regex/pattern-based section detection for fallback parsing."""

    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp_available = True
        except (OSError, ImportError):
            self.nlp_available = False
        
        # Universal section patterns that work across all domains
        self.universal_section_patterns = {
            "name": [
                r"name", r"full\s+name", r"first\s+name", r"last\s+name", r"given\s+name"
            ],
            "education": [
                r"education", r"academic\s+background", r"academics", 
                r"academic\s+history", r"qualifications", r"academic\s+credentials",
                r"degrees", r"academic\s+achievements", r"scholastic", r"learning"
            ],
            "experience": [
                r"experience", r"work\s+history", r"professional\s+experience", 
                r"employment", r"work\s+experience", r"career\s+history",
                r"professional\s+background", r"work\s+background", r"employment\s+history",
                r"work\s+record", r"professional\s+journey"
            ],
            "skills": [
                r"skills", r"competencies", r"expertise", r"capabilities", 
                r"proficiencies", r"abilities", r"qualifications", r"competences",
                r"strengths", r"proficiencies", r"skill\s+set"
            ],
            "projects": [
                r"projects", r"selected\s+projects", r"personal\s+projects", 
                r"academic\s+projects", r"key\s+projects", r"project\s+experience",
                r"portfolio", r"project\s+work", r"achievements", r"key\s+achievements",
                r"notable\s+projects", r"project\s+portfolio"
            ],
            "certifications": [
                r"certifications", r"certificates", r"professional\s+certifications",
                r"accreditations", r"licenses", r"professional\s+credentials",
                r"industry\s+certifications", r"training\s+certificates", r"licenses",
                r"professional\s+licenses", r"accreditations", r"credentials"
            ],
            "publications": [
                r"publications", r"research", r"papers", r"articles", r"journals", 
                r"conferences", r"research\s+papers", r"academic\s+papers", 
                r"published\s+work", r"research\s+publications", r"academic\s+output"
            ]
        }
    

    def read_pdf(self, file_path: str) -> Optional[str]:
        """Read PDF file content."""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None
    
    def read_docx(self, file_path: str) -> Optional[str]:
        """Read DOCX file content."""
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return None
        
    def read_file(self, file_path: str) -> Optional[str]:
        """Read file content based on file type."""
        
        file_extension = file_path.split('.')[-1].lower()
        
        try:
            if file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_extension == 'pdf':
                return self.read_pdf(file_path)
            elif file_extension == 'docx':
                return self.read_docx(file_path)
            else:
                return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    
    def extract_basic_info(self, content: str) -> Dict[str, Any]:
        """Extract basic information without domain assumptions."""
        
        lines = content.split('\n')
        basic_info = {
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
        }
        
        # Extract name (usually first non-empty line)
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 2 and not line.startswith(('•', '-', '*', '#')):
                # Simple heuristic: name is usually 2-4 words, starts with capital
                words = line.split()
                if 2 <= len(words) <= 4 and line[0].isupper():
                    basic_info["name"] = line
                    break
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, content)
        if email_match:
            basic_info["email"] = email_match.group()
        
        # Extract phone
        phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US format
            r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b'  # International
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, content)
            if phone_match:
                basic_info["phone"] = phone_match.group()
                break
        
        # Extract location (look for city, state patterns)
        location_pattern = r'\b[A-Z][a-z]+(?:[\s,]+[A-Z]{2})?\s*\d{5}\b'  # City, State ZIP
        location_match = re.search(location_pattern, content)
        if location_match:
            basic_info["location"] = location_match.group()
        
        return basic_info
    

    def find_section_boundaries(self, lines: List[str]) -> Dict[str, Tuple[int, int]]:
        """Find start and end positions of each section using universal patterns."""
        
        boundaries = {section: (-1, -1) for section in self.universal_section_patterns.keys()}
        section_starts = {}  # Track where each section starts
        
        # First pass: find all section headers
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if not line_lower:
                continue
            
            # Check each section pattern
            for section_name, patterns in self.universal_section_patterns.items():
                if section_name in section_starts:  # Already found this section
                    continue
                    
                for pattern in patterns:
                    if re.search(f'^{pattern}$', line_lower, re.IGNORECASE):
                        section_starts[section_name] = i + 1  # Start after header
                        break
        
        # Second pass: determine section boundaries
        section_order = []
        for line_num in sorted(section_starts.values()):
            for section_name, start_line in section_starts.items():
                if start_line == line_num:
                    section_order.append((section_name, start_line))
                    break
        
        # Set boundaries based on order
        for i, (section_name, start) in enumerate(section_order):
            if i < len(section_order) - 1:
                # End at the start of the next section
                _, next_start = section_order[i + 1]
                end = next_start - 1  # -1 because next_start is after the next header
            else:
                # Last section goes to end of document
                end = len(lines)
            
            boundaries[section_name] = (start, end)
        
        return boundaries
    

    def parse_section_content_universal(self, section_name: str, content_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse content of a specific section without domain assumptions."""
        
        if section_name == "experience":
            return self._parse_experience_universal(content_lines)
        elif section_name == "education":
            return self._parse_education_universal(content_lines)
        elif section_name == "projects":
            return self._parse_projects_universal(content_lines)
        elif section_name == "skills":
            return self._parse_skills_universal(content_lines)
        elif section_name == "certifications":
            return self._parse_certifications_universal(content_lines)
        elif section_name == "publications":
            return self._parse_publications_universal(content_lines)
        else:
            return []
    
    def _parse_experience_universal(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse work experience section without domain assumptions."""
        
        experiences = []
        current_exp = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for job title patterns (universal across domains)
            if re.search(r'^(senior|junior|lead|principal|staff|senior\s+)?([a-z\s]+)?(engineer|developer|scientist|analyst|manager|consultant|specialist|coordinator|assistant|director|supervisor|technician|technologist|researcher|instructor|professor|lecturer|coordinator|administrator|officer|representative|associate|executive|president|vice\s+president|ceo|cfo|cto|coo)', line, re.IGNORECASE):
                if current_exp:
                    experiences.append(current_exp)
                current_exp = {"title": line, "company": "", "duration": "", "description": []}
            
            # Look for company names (universal pattern)
            elif current_exp and not current_exp["company"] and re.search(r'at\s+([^,]+)', line, re.IGNORECASE):
                company_match = re.search(r'at\s+([^,]+)', line, re.IGNORECASE)
                if company_match:
                    current_exp["company"] = company_match.group(1).strip()
            
            # Look for duration patterns (universal)
            elif current_exp and not current_exp["duration"] and re.search(r'\d{4}\s*[-–]\s*(present|\d{4})', line, re.IGNORECASE):
                current_exp["duration"] = line
            
            # Add description lines
            elif current_exp and line and not line.startswith(('•', '-', '*', '#')):
                current_exp["description"].append(line)
        
        # Add last experience
        if current_exp:
            experiences.append(current_exp)
        
        return experiences
    
    def _parse_education_universal(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse education section without domain assumptions."""
        
        education = []
        current_edu = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for degree patterns (universal across domains)
            if re.search(r'(bachelor|master|phd|doctorate|associate|diploma|certificate|bachelor\s+of|master\s+of|doctor\s+of|associate\s+of|diploma\s+in|certificate\s+in)', line, re.IGNORECASE):
                if current_edu:
                    education.append(current_edu)
                current_edu = {"degree": line, "institution": "", "duration": "", "field": ""}
            
            # Look for institution names
            elif current_edu and not current_edu["institution"] and re.search(r'from\s+([^,]+)', line, re.IGNORECASE):
                institution_match = re.search(r'from\s+([^,]+)', line, re.IGNORECASE)
                if institution_match:
                    current_edu["institution"] = institution_match.group(1).strip()
            
            # Look for duration
            elif current_edu and not current_edu["duration"] and re.search(r'\d{4}\s*[-–]\s*(present|\d{4})', line, re.IGNORECASE):
                current_edu["duration"] = line
        
        # Add last education
        if current_edu:
            education.append(current_edu)
        
        return education
    
    def _parse_projects_universal(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse projects section without domain assumptions."""
        
        projects = []
        current_project = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for project title patterns (universal)
            if re.search(r'^[A-Z][^:]+(?:[:]|$)', line) and len(line) < 100:
                if current_project:
                    projects.append(current_project)
                current_project = {"title": line.replace(':', '').strip(), "description": [], "technologies": []}
            
            # Add description
            elif current_project and line:
                current_project["description"].append(line)
        
        # Add last project
        if current_project:
            projects.append(current_project)
        
        return projects
    
    def _parse_skills_universal(self, lines: List[str]) -> List[str]:
        """Parse skills section without domain assumptions."""
        
        skills = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split by common delimiters (universal)
            skill_parts = re.split(r'[,•\-\*\|]', line)
            for skill in skill_parts:
                skill = skill.strip()
                if skill and len(skill) > 1:
                    # Add if it looks like a skill (not too long, not all caps)
                    if len(skill) < 50 and not skill.isupper():
                        skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    def _parse_certifications_universal(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse certifications section without domain assumptions."""
        
        certifications = []
        current_cert = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for certification name patterns (universal)
            if re.search(r'^(certified|certification|certificate)', line, re.IGNORECASE) or re.search(r'([A-Z][^:]+(?:[:]|$))', line):
                if current_cert:
                    certifications.append(current_cert)
                
                cert_name = line.replace(':', '').strip()
                current_cert = {"name": cert_name, "issuer": "", "date": ""}
            
            # Look for issuer
            elif current_cert and not current_cert["issuer"] and re.search(r'by\s+([^,]+)', line, re.IGNORECASE):
                issuer_match = re.search(r'by\s+([^,]+)', line, re.IGNORECASE)
                if issuer_match:
                    current_cert["issuer"] = issuer_match.group(1).strip()
            
            # Look for date
            elif current_cert and not current_cert["date"] and re.search(r'\d{4}', line):
                current_cert["date"] = line
        
        # Add last certification
        if current_cert:
            certifications.append(current_cert)
        
        return certifications
    
    def _parse_publications_universal(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse publications section without domain assumptions."""
        
        publications = []
        current_pub = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for publication title patterns (universal)
            if re.search(r'^[A-Z][^:]+(?:[:]|$)', line) and len(line) < 150:
                if current_pub:
                    publications.append(current_pub)
                
                pub_title = line.replace(':', '').strip()
                current_pub = {"title": pub_title, "authors": "", "journal": "", "year": ""}
            
            # Add description
            elif current_pub and line:
                current_pub["description"].append(line)
        
        # Add last publication
        if current_pub:
            publications.append(current_pub)
        
        return publications
    

    def get_parsing_summary(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of parsed resume data."""
        
        if "error" in parsed_data:
            return {"error": parsed_data["error"]}
        
        summary = {
            "total_sections": 0,
            "sections_found": [],
            "basic_info_complete": False,
            "parsing_quality": "unknown"
        }
        
        # Count sections
        for section in self.universal_section_patterns.keys():
            if section in parsed_data and parsed_data[section]:
                summary["total_sections"] += 1
                summary["sections_found"].append(section)
        
        # Check basic info completeness
        basic_info = parsed_data.get("basic_info", {})
        required_fields = ["name", "email"]
        summary["basic_info_complete"] = all(basic_info.get(field) for field in required_fields)
        
        # Assess parsing quality
        if summary["total_sections"] >= 4 and summary["basic_info_complete"]:
            summary["parsing_quality"] = "excellent"
        elif summary["total_sections"] >= 3 and summary["basic_info_complete"]:
            summary["parsing_quality"] = "good"
        elif summary["total_sections"] >= 2:
            summary["parsing_quality"] = "fair"
        else:
            summary["parsing_quality"] = "poor"
        
        return summary
    
    def extract_sections(self, content: str) -> Dict[str, Any]:
        """Extract all sections using universal patterns."""
        
        sections = {}
        
        # Split content into lines for processing
        lines = content.split('\n')
        
        # Find section boundaries
        section_boundaries = self.find_section_boundaries(lines)

        # Extract each section
        for section_name, (start, end) in section_boundaries.items():
            if start != -1 and end != -1:
                section_content = lines[start:end]
                sections[section_name] = self.parse_section_content_universal(section_name, section_content)
        
        return sections
    

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Parse resume using regex/pattern-based approach only."""
        try:
            content = self.read_file(file_path)
            if not content:
                return {"error": "Failed to read file"}
            
            basic_info = self.extract_basic_info(content)
            sections = self.extract_sections(content)

            parsed_resume = {
                "basic_info": basic_info,
                **sections
            }

            return parsed_resume

        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}


def main():
    """Test the resume parser."""
    parser = ResumeParser()

    print("Testing Resume Parser...")
    resume_path = "JaySGoenka Resume.pdf"
    print(f"Parsing resume: {resume_path}")
    parsed_data = parser.parse_resume(resume_path)
    
    if "error" not in parsed_data:
        print("Resume parsed successfully!")
        print("\nParsed Sections Only:")

        # Format sections to match intelligent parser output
        section_keys = ["education", "skills", "experience", "projects", "certifications", "publications"]
        formatted_sections = {}
        
        for key in section_keys:
            if key in parsed_data and parsed_data[key]:
                # Convert parsed data to string format for consistency
                if isinstance(parsed_data[key], list):
                    if key == "skills":
                        # Skills are just a list of strings
                        formatted_sections[key] = "\n".join(parsed_data[key])
                    else:
                        # Other sections are list of dicts, format them nicely
                        formatted_content = []
                        for item in parsed_data[key]:
                            if isinstance(item, dict):
                                # Format dict items as readable text
                                item_text = []
                                for field, value in item.items():
                                    if value:
                                        if isinstance(value, list):
                                            item_text.append(f"{field.title()}: {', '.join(value)}")
                                        else:
                                            item_text.append(f"{field.title()}: {value}")
                                formatted_content.append(" | ".join(item_text))
                            else:
                                formatted_content.append(str(item))
                        formatted_sections[key] = "\n".join(formatted_content)
                else:
                    formatted_sections[key] = str(parsed_data[key])
        
        # Include basic info and sections for LLM
        output_data = {
            "basic_info": parsed_data.get("basic_info", {}),
            "sections": formatted_sections
        }
        
        print(json.dumps(output_data, indent=2, ensure_ascii=False))
        with open("parsed_resume.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            print("\nFiltered sections and basic info saved to 'parsed_resume.json'")
    else:
        print(f"Parsing failed: {parsed_data['error']}")


if __name__ == "__main__":
    main()
