import re
import json
import spacy
from typing import Dict, List, Any, Optional, Tuple
import os
import PyPDF2
from docx import Document


class ResumeParser:
    """Smart resume parser using NLP to extract sections with header variations."""
    
    def __init__(self):
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.nlp_available = True
        except OSError:
            print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp_available = False
        
        # section header variations
        self.section_patterns = {
            "experience": [
                r"experience", r"work\s+history", r"professional\s+experience", 
                r"employment", r"work\s+experience", r"career\s+history",
                r"professional\s+background", r"work\s+background"
            ],
            "education": [
                r"education", r"academic\s+background", r"academics", 
                r"academic\s+history", r"qualifications", r"academic\s+credentials",
                r"degrees", r"academic\s+achievements"
            ],
            "projects": [
                r"projects", r"selected\s+projects", r"personal\s+projects", 
                r"academic\s+projects", r"key\s+projects", r"project\s+experience",
                r"portfolio", r"project\s+work"
            ],
            "skills": [
                r"skills", r"technical\s+skills", r"tech\s+stack", r"core\s+competencies",
                r"technical\s+expertise", r"competencies", r"expertise", r"technologies",
                r"programming\s+languages", r"tools", r"frameworks"
            ],
            "certifications": [
                r"certifications", r"certificates", r"professional\s+certifications",
                r"accreditations", r"licenses", r"professional\s+credentials",
                r"industry\s+certifications", r"training\s+certificates"
            ]
        }
        
        # Common skill keywords for extraction
        self.skill_keywords = {
            "programming": ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "swift", "kotlin"],
            "web_tech": ["html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "spring"],
            "databases": ["sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "dynamodb"],
            "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "gitlab"],
            "ml_ai": ["machine learning", "deep learning", "ai", "tensorflow", "pytorch", "scikit-learn", "nlp", "computer vision"],
            "data": ["data science", "data analysis", "pandas", "numpy", "matplotlib", "seaborn", "tableau", "powerbi"],
            "tools": ["git", "jira", "confluence", "slack", "trello", "asana", "figma", "adobe"]
        }
    

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Main method to parse resume and extract all sections."""
        
        try:
            content = self._read_file(file_path)
            if not content:
                return {"error": "Failed to read file"}
            
            basic_info = self._extract_basic_info(content)

            sections = self._extract_sections(content)
            
            parsed_resume = {
                "basic_info": basic_info,
                **sections
            }
            
            return parsed_resume
            
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}
    

    def _read_file(self, file_path: str) -> Optional[str]:
        """Read file content based on file type."""
        
        file_extension = file_path.split('.')[-1].lower()
        
        try:
            if file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_extension == 'pdf':
                return self._read_pdf(file_path)
            elif file_extension == 'docx':
                return self._read_docx(file_path)
            else:
                return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def _read_pdf(self, file_path: str) -> Optional[str]:
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
    
    def _read_docx(self, file_path: str) -> Optional[str]:
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
    
    def _extract_basic_info(self, content: str) -> Dict[str, Any]:
        """Extract basic information like name, contact, summary."""
        
        lines = content.split('\n')
        basic_info = {
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "summary": ""
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
        
        # Extract summary (look for longer paragraph after name)
        summary_start = -1
        for i, line in enumerate(lines):
            if basic_info["name"] in line:
                summary_start = i + 1
                break
        
        if summary_start > 0 and summary_start < len(lines):
            summary_lines = []
            for line in lines[summary_start:summary_start + 5]:  # Check next 5 lines
                line = line.strip()
                if line and len(line) > 20:  # Summary lines are usually longer
                    summary_lines.append(line)
                elif line and len(line) < 10:  # Short lines might be section headers
                    break
            if summary_lines:
                basic_info["summary"] = " ".join(summary_lines)
        
        return basic_info
    
    def _extract_sections(self, content: str) -> Dict[str, Any]:
        """Extract all sections using NLP and pattern matching."""
        sections = {}
        
        # Split content into lines for processing
        lines = content.split('\n')
        
        # Find section boundaries
        section_boundaries = self._find_section_boundaries(lines)
        
        # Extract each section
        for section_name, (start, end) in section_boundaries.items():
            if start != -1 and end != -1:
                section_content = lines[start:end]
                sections[section_name] = self._parse_section_content(section_name, section_content)
        
        return sections
    
    def _find_section_boundaries(self, lines: List[str]) -> Dict[str, Tuple[int, int]]:
        """Find start and end positions of each section."""
        boundaries = {section: (-1, -1) for section in self.section_patterns.keys()}
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check each section pattern
            for section_name, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line_lower, re.IGNORECASE):
                        # Found section header
                        start = i + 1  # Start after header
                        
                        # Find end of section (next section header or end of content)
                        end = len(lines)
                        for j in range(i + 1, len(lines)):
                            next_line = lines[j].lower().strip()
                            # Check if next line is another section header
                            for other_section, other_patterns in self.section_patterns.items():
                                if other_section != section_name:
                                    for other_pattern in other_patterns:
                                        if re.search(other_pattern, next_line, re.IGNORECASE):
                                            end = j
                                            break
                                    if end != len(lines):
                                        break
                            if end != len(lines):
                                break
                        
                        boundaries[section_name] = (start, end)
                        break
                if boundaries[section_name][0] != -1:
                    break
        
        return boundaries
    
    def _parse_section_content(self, section_name: str, content_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse content of a specific section."""
        if section_name == "experience":
            return self._parse_experience(content_lines)
        elif section_name == "education":
            return self._parse_education(content_lines)
        elif section_name == "projects":
            return self._parse_projects(content_lines)
        elif section_name == "skills":
            return self._parse_skills(content_lines)
        elif section_name == "certifications":
            return self._parse_certifications(content_lines)
        else:
            return []
    
    def _parse_experience(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse work experience section."""
        experiences = []
        current_exp = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for job title patterns
            if re.search(r'^(senior|junior|lead|principal|staff|senior\s+)?(software\s+)?(engineer|developer|scientist|analyst|manager|consultant)', line, re.IGNORECASE):
                if current_exp:
                    experiences.append(current_exp)
                current_exp = {"title": line, "company": "", "duration": "", "description": []}
            
            # Look for company names (usually after title, before duration)
            elif current_exp and not current_exp["company"] and re.search(r'at\s+([^,]+)', line, re.IGNORECASE):
                company_match = re.search(r'at\s+([^,]+)', line, re.IGNORECASE)
                if company_match:
                    current_exp["company"] = company_match.group(1).strip()
            
            # Look for duration patterns
            elif current_exp and not current_exp["duration"] and re.search(r'\d{4}\s*[-–]\s*(present|\d{4})', line, re.IGNORECASE):
                current_exp["duration"] = line
            
            # Add description lines
            elif current_exp and line and not line.startswith(('•', '-', '*', '#')):
                current_exp["description"].append(line)
        
        # Add last experience
        if current_exp:
            experiences.append(current_exp)
        
        return experiences
    
    def _parse_education(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse education section."""
        education = []
        current_edu = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for degree patterns
            if re.search(r'(bachelor|master|phd|doctorate|associate|diploma)', line, re.IGNORECASE):
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
    
    def _parse_projects(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse projects section."""
        projects = []
        current_project = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for project title patterns
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
    
    def _parse_skills(self, lines: List[str]) -> List[str]:
        """Parse skills section."""
        skills = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split by common delimiters
            skill_parts = re.split(r'[,•\-\*\|]', line)
            for skill in skill_parts:
                skill = skill.strip()
                if skill and len(skill) > 1:
                    # Check if it's a recognized skill
                    for category, skill_list in self.skill_keywords.items():
                        if skill.lower() in skill_list or any(skill.lower() in s.lower() for s in skill_list):
                            skills.append(skill)
                            break
                    else:
                        # Add if it looks like a skill (not too long, not all caps)
                        if len(skill) < 30 and not skill.isupper():
                            skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    def _parse_certifications(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse certifications section."""
        certifications = []
        current_cert = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for certification name patterns
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
        for section in self.section_patterns.keys():
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


def main():
    """Test the smart resume parser."""
    parser = SmartResumeParser()
    
    print("Testing Smart Resume Parser...")
    
    # Test with a sample resume text
    sample_resume = """
    John Doe
    Software Engineer
    john.doe@email.com
    (555) 123-4567
    San Francisco, CA
    
    SUMMARY
    Experienced software engineer with 5+ years in full-stack development.
    
    PROFESSIONAL EXPERIENCE
    Senior Software Engineer at TechCorp
    2020 - Present
    • Led development of microservices architecture
    • Mentored junior developers
    
    Software Developer at StartupXYZ
    2018 - 2020
    • Built REST APIs using Node.js
    • Implemented CI/CD pipelines
    
    EDUCATION
    Bachelor of Science in Computer Science from University of California
    2014 - 2018
    
    TECHNICAL SKILLS
    Python, JavaScript, React, Node.js, AWS, Docker, Kubernetes
    
    PROJECTS
    E-commerce Platform: Built full-stack application using React and Node.js
    Machine Learning Model: Developed predictive analytics model using Python
    
    CERTIFICATIONS
    AWS Certified Developer Associate
    Certified Scrum Master
    """
    
    # Save sample resume to file for testing
    with open("sample_resume.txt", "w", encoding="utf-8") as f:
        f.write(sample_resume)
    
    print("Sample resume created: sample_resume.txt")
    
    # Parse the resume
    print("\nParsing resume...")
    parsed_data = parser.parse_resume("sample_resume.txt")
    
    if "error" not in parsed_data:
        print("Resume parsed successfully!")
        
        # Display parsed data
        print("\nParsed Resume Data:")
        print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
        
        # Get parsing summary
        summary = parser.get_parsing_summary(parsed_data)
        print(f"\nParsing Summary: {summary}")
        
        # Save parsed data
        with open("parsed_resume.json", "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2, ensure_ascii=False)
        print("\nParsed data saved to 'parsed_resume.json'")
        
    else:
        print(f"Parsing failed: {parsed_data['error']}")
    
    # Clean up
    if os.path.exists("sample_resume.txt"):
        os.remove("sample_resume.txt")


if __name__ == "__main__":
    main()
