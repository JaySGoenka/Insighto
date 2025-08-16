import json
import os
from profile_extractor import LinkedInProfileExtractor
from resume_parser import ResumeParser
from typing import Dict, Any, Optional
from langchain_ollama import OllamaLLM
from langchain.prompts.prompt import PromptTemplate

class ComprehensiveProfileProcessor:
    """Processes LinkedIn profiles and resumes to generate comprehensive candidate reports."""
    
    def __init__(self):
        self.extractor = LinkedInProfileExtractor()
        self.resume_parser = ResumeParser()
        
        try:
            self.llm = OllamaLLM(model="llama3.1")
            self.llm_available = True
        except Exception as e:
            print(f"Warning: LLM not available: {e}")
            self.llm_available = False
    

    def _format_linkedin_data(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Format LinkedIn data for comprehensive analysis."""
        
        formatted = {
            "source": "LinkedIn Profile",
            "basic_info": {
                "name": f"{profile.get('basic_info', {}).get('firstName', '')} {profile.get('basic_info', {}).get('lastName', '')}",
                "headline": profile.get('basic_info', {}).get('headline', ''),
                "location": profile.get('basic_info', {}).get('location', ''),
                "about": profile.get('basic_info', {}).get('about', '')
            },
            "experience": profile.get('experience', []),
            "skills": profile.get('skills', []),
            "education": profile.get('education', []),
            "projects": profile.get('projects', []),
            "recommendations": profile.get('recommendations', [])
        }
        return formatted


    def process_linkedin_profile(self, linkedin_url: str) -> Optional[Dict[str, Any]]:
        """Extract and process LinkedIn profile data."""
        
        try:
            extracted_profile = self.extractor.process_profile(linkedin_url)
            if extracted_profile:
                return self._format_linkedin_data(extracted_profile)
            return None
        except Exception as e:
            print(f"Error processing LinkedIn profile: {e}")
            return None
    
    def _format_resume_data(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format resume data for comprehensive analysis."""
        
        if "error" in resume_data:
            return resume_data
        
        # The smart parser already provides well-structured data
        formatted = {
            "source": "Resume",
            "basic_info": resume_data.get("basic_info", {}),
            "experience": resume_data.get("experience", []),
            "skills": resume_data.get("skills", []),
            "education": resume_data.get("education", []),
            "projects": resume_data.get("projects", []),
            "certifications": resume_data.get("certifications", [])
        }
        return formatted


    def process_resume_file(self, resume_file_path: str) -> Optional[Dict[str, Any]]:
        """Process resume file and extract relevant information using smart parser."""
        
        try:
            parsed_resume = self.resume_parser.parse_resume(resume_file_path)
            if "error" in parsed_resume:
                print(f"Resume parsing error: {parsed_resume['error']}")
                return None
            return parsed_resume
        except Exception as e:
            print(f"Error processing resume: {e}")
            return None
    

    def _create_comprehensive_prompt(self, linkedin_data: Dict[str, Any], resume_data: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for LLM analysis."""
        
        prompt = f"""
        You are an expert HR professional tasked with creating a comprehensive candidate report.
        Please analyze the following LinkedIn profile and resume data to generate a detailed, professional report.
        
        LINKEDIN PROFILE DATA:
        {json.dumps(linkedin_data, indent=2, ensure_ascii=False)}
        
        RESUME DATA:
        {json.dumps(resume_data, indent=2, ensure_ascii=False)}
        
        Please provide a comprehensive candidate report that includes:
        
        1. **EXECUTIVE SUMMARY** (2-3 sentences)
           - Overall professional assessment
           - Key qualifications and strengths
        
        2. **PROFESSIONAL BACKGROUND**
           - Career progression and experience
           - Notable achievements and responsibilities
           - Industry expertise and domain knowledge
        
        3. **TECHNICAL SKILLS & EXPERTISE**
           - Core technical competencies
           - Programming languages and technologies
           - Tools and platforms experience
           - Certifications and training
        
        4. **EDUCATION & QUALIFICATIONS**
           - Academic background
           - Relevant coursework and projects
           - Professional development
        
        5. **KEY PROJECTS & ACCOMPLISHMENTS**
           - Significant projects and their impact
           - Innovation and problem-solving examples
           - Quantifiable achievements
        
        6. **INTERPERSONAL & LEADERSHIP SKILLS**
           - Team collaboration experience
           - Leadership roles and responsibilities
           - Communication and presentation abilities
        
        7. **CAREER RECOMMENDATIONS**
           - Suitable roles and industries
           - Growth potential and development areas
           - Market positioning
        
        8. **OVERALL ASSESSMENT**
           - Strengths and competitive advantages
           - Areas for improvement
           - Fit for different types of opportunities
        
        Please write this report in a professional, objective tone suitable for HR professionals and hiring managers.
        Focus on actionable insights and clear, structured information.
        """
        
        return prompt


    def generate_comprehensive_report(self, linkedin_data: Dict[str, Any], resume_data: Dict[str, Any]) -> str:
        """Generate a comprehensive candidate report using LLM."""
        
        if not self.llm_available:
            return self._generate_fallback_report(linkedin_data, resume_data)
        
        try:
            prompt = self._create_comprehensive_prompt(linkedin_data, resume_data)
            response = self.llm.invoke(prompt)
            return response if response else "Report generation failed."
            
        except Exception as e:
            print(f"Error generating LLM report: {e}")
            return self._generate_fallback_report(linkedin_data, resume_data)
    
    
    
    def _generate_fallback_report(self, linkedin_data: Dict[str, Any], resume_data: Dict[str, Any]) -> str:
        """Generate a fallback report when LLM is not available."""
        
        report = []
        
        # Basic information
        linkedin_name = linkedin_data.get('basic_info', {}).get('name', 'N/A')
        resume_name = resume_data.get('basic_info', {}).get('name', 'N/A')
        name = linkedin_name if linkedin_name != 'N/A' else resume_name
        
        report.append(f"COMPREHENSIVE CANDIDATE REPORT")
        report.append(f"Generated without LLM assistance")
        report.append("=" * 50)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append(f"Candidate: {name}")
        report.append(f"LinkedIn Headline: {linkedin_data.get('basic_info', {}).get('headline', 'N/A')}")
        report.append(f"Location: {linkedin_data.get('basic_info', {}).get('location', 'N/A')}")
        report.append("")
        
        # Experience Summary
        experiences = linkedin_data.get('experience', [])
        if experiences:
            report.append("PROFESSIONAL EXPERIENCE")
            report.append(f"Total positions: {len(experiences)}")
            for i, exp in enumerate(experiences[:3], 1):
                report.append(f"{i}. {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")
                report.append(f"   Duration: {exp.get('duration', 'N/A')}")
            report.append("")
        
        # Skills Summary
        skills = linkedin_data.get('skills', [])
        if skills:
            report.append("TECHNICAL SKILLS")
            skill_names = [skill.get('skill', '') for skill in skills[:10]]
            report.append(f"Top skills: {', '.join(skill_names)}")
            report.append("")
        
        # Education Summary
        education = linkedin_data.get('education', [])
        if education:
            report.append("EDUCATION")
            for edu in education:
                report.append(f"- {edu.get('degree', 'N/A')} from {edu.get('institution', 'N/A')}")
            report.append("")
        
        # Resume Summary
        if resume_data and 'error' not in resume_data:
            report.append("RESUME INFORMATION")
            report.append(f"Resume name: {resume_data.get('basic_info', {}).get('name', 'N/A')}")
            resume_skills = resume_data.get('skills', [])
            if resume_skills:
                report.append(f"Resume skills: {', '.join(resume_skills)}")
            report.append("")
        
        report.append("NOTE: This is a basic report generated without AI assistance.")
        report.append("For a more comprehensive analysis, ensure LLM services are available.")
        
        return "\n".join(report)
    
    
    def get_candidate_summary(self, linkedin_data: Dict[str, Any], resume_data: Dict[str, Any]) -> Dict[str, Any]:
        
        summary = {
            "name": linkedin_data.get('basic_info', {}).get('name', 'N/A'),
            "headline": linkedin_data.get('basic_info', {}).get('headline', 'N/A'),
            "location": linkedin_data.get('basic_info', {}).get('location', 'N/A'),
            "total_experience": len(linkedin_data.get('experience', [])),
            "total_skills": len(linkedin_data.get('skills', [])),
            "total_education": len(linkedin_data.get('education', [])),
            "total_projects": len(linkedin_data.get('projects', [])),
            "has_resume": 'error' not in resume_data if resume_data else False,
            "llm_available": self.llm_available
        }
        return summary


def main():
    """Test the comprehensive profile processor."""
    
    processor = ComprehensiveProfileProcessor()
    
    print("Testing Comprehensive Profile Processor...")
    
    # Test LinkedIn profile processing
    linkedin_url = "https://www.linkedin.com/in/jay-goenka-b797851b2/"
    print(f"\nProcessing LinkedIn profile: {linkedin_url}")
    
    linkedin_data = processor.process_linkedin_profile(linkedin_url)
    if linkedin_data:
        print("LinkedIn profile processed successfully")
    else:
        print("Failed to process LinkedIn profile")
        return
    
    resume_data = None

    # Generate comprehensive report
    print("\n Generating comprehensive candidate report...")
    report = processor.generate_comprehensive_report(linkedin_data, resume_data)
    
    if report:
        print("Report generated successfully!")
        print("\n" + "="*60)
        print("COMPREHENSIVE CANDIDATE REPORT")
        print("="*60)
        print(report)
        
        # Save report to file
        with open("comprehensive_candidate_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n Report saved to 'comprehensive_candidate_report.txt'")
        
        # Get summary
        summary = processor.get_candidate_summary(linkedin_data, resume_data)
        print(f"\n Candidate Summary: {summary}")
        
    else:
        print(" Failed to generate report")


if __name__ == "__main__":
    main()
