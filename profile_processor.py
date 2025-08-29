import json
import os
from profile_extractor import LinkedInProfileExtractor
from resume_parser import ResumeParser
from intelligent_resume_parser import IntelligentResumeParser
from typing import Dict, Any, Optional
from langchain_ollama import OllamaLLM

class ComprehensiveProfileProcessor:
    """Streamlined processor for LinkedIn profiles and resumes."""
    
    def __init__(self):
        self.linkedin_extractor = LinkedInProfileExtractor()
        self.resume_parser = ResumeParser()
        self.intelligent_parser = IntelligentResumeParser()
        
        try:
            self.llm = OllamaLLM(model="llama3.1")
            self.llm_available = True
        except Exception as e:
            print(f"Warning: LLM not available: {e}")
            self.llm_available = False


    def process_linkedin_profile(self, linkedin_url: str) -> Optional[Dict[str, Any]]:
        """Extract LinkedIn profile data."""
        
        try:
            return self.linkedin_extractor.process_profile(linkedin_url)
        except Exception as e:
            print(f"Error processing LinkedIn profile: {e}")
            return None


    def process_resume_file(self, resume_file_path: str) -> Optional[Dict[str, Any]]:
        """Process resume file with intelligent parser first, fallback to regex parser."""
        
        try:
            # Try intelligent parser first
            parsed_intelligent = self.intelligent_parser.parse_resume(resume_file_path)
            
            if "error" not in parsed_intelligent:
                confidence = parsed_intelligent.get("metadata", {}).get("confidence_score", 0.0)
                print(f"Intelligent parser confidence: {confidence}")
                
                if confidence >= 0.4:
                    return {"data": parsed_intelligent, "parser_used": "intelligent"}
            
            # Fallback to regex parser
            print("Using fallback regex parser...")
            parsed_regex = self.resume_parser.parse_resume(resume_file_path)
            
            if "error" in parsed_regex:
                print(f"Resume parsing error: {parsed_regex['error']}")
                return None
            
            return {"data": parsed_regex, "parser_used": "regex"}
            
        except Exception as e:
            print(f"Error processing resume: {e}")
            return None
        

    def _create_prompt(self, linkedin_data: Dict[str, Any], resume_data: Dict[str, Any]) -> str:
        """Create a simple, focused prompt for LLM analysis."""
        
        data_sources = []
        
        if linkedin_data:
            data_sources.append(f"LINKEDIN PROFILE:\n{json.dumps(linkedin_data, indent=2)}")
        
        if resume_data:
            data_sources.append(f"RESUME DATA:\n{json.dumps(resume_data, indent=2)}")
        
        combined_data = "\n\n".join(data_sources)
        
        return f"""Analyze the following candidate information and provide a comprehensive but concise report for recruiters:

{combined_data}

Please provide a professional assessment that covers the candidate's background, skills, experience, and suitability for roles. Keep the response focused and actionable for hiring decisions."""


    def generate_report(self, linkedin_data: Dict[str, Any] = None, resume_data: Dict[str, Any] = None) -> str:
        """Generate candidate report using available data."""
        
        if not linkedin_data and not resume_data:
            return "No data available to generate report."
        
        if not self.llm_available:
            return self._generate_simple_fallback(linkedin_data, resume_data)
        
        try:
            prompt = self._create_simple_prompt(linkedin_data, resume_data)
            response = self.llm.invoke(prompt)
            return response if response else "Report generation failed."
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return self._generate_simple_fallback(linkedin_data, resume_data)


    def _generate_simple_fallback(self, linkedin_data: Dict[str, Any], resume_data: Dict[str, Any]) -> str:
        """Generate a simple fallback report."""
        
        report = ["CANDIDATE REPORT", "=" * 40, ""]
        
        # Basic info from LinkedIn or resume
        if linkedin_data:
            basic = linkedin_data.get('basic_info', {})
            name = f"{basic.get('firstName', '')} {basic.get('lastName', '')}".strip()
            if name:
                report.append(f"Name: {name}")
            if basic.get('headline'):
                report.append(f"Title: {basic.get('headline')}")
            if basic.get('location'):
                report.append(f"Location: {basic.get('location')}")
            
            # Experience count
            experiences = linkedin_data.get('experience', [])
            if experiences:
                report.append(f"Experience: {len(experiences)} positions")
            
            # Skills count
            skills = linkedin_data.get('skills', [])
            if skills:
                report.append(f"Skills: {len(skills)} listed")
        
        elif resume_data:
            basic = resume_data.get('basic_info', {})
            if basic.get('name'):
                report.append(f"Name: {basic.get('name')}")
            if basic.get('email'):
                report.append(f"Email: {basic.get('email')}")
        
        report.append("")
        report.append("Note: Limited analysis available without LLM.")
        
        return "\n".join(report)

def main():
    """Test the streamlined profile processor."""
    processor = ComprehensiveProfileProcessor()
    
    # Test with LinkedIn URL
    linkedin_url = "https://www.linkedin.com/in/jay-goenka-b797851b2/"
    linkedin_data = processor.process_linkedin_profile(linkedin_url)
    
    resume_data = None
    
    # Generate report
    report = processor.generate_report(linkedin_data, resume_data)
    print("CANDIDATE REPORT")
    print("=" * 50)
    print(report)

if __name__ == "__main__":
    main()
