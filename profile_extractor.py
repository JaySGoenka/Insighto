import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

class LinkedInProfileExtractor:
    """Extracts and structures relevant LinkedIn profile information for LLM processing."""
    
    def __init__(self):
        load_dotenv()
    
    def validate_linkedin_url(self, url: str) -> bool:
        """Validate if the provided URL is a valid LinkedIn profile URL."""
        
        if not url:
            return False
        
        # Basic validation for LinkedIn profile URLs
        valid_patterns = [
            "linkedin.com/in/",
            "www.linkedin.com/in/",
            "https://linkedin.com/in/",
            "https://www.linkedin.com/in/"
        ]
        
        return any(pattern in url for pattern in valid_patterns)
    
    def fetch_linkedin_profile(self, linkedin_url: str) -> Optional[Dict[str, Any]]:
        """Fetch LinkedIn profile data from Apify API."""
        
        APIFY_TOKEN = os.getenv("APIFY_TOKEN")
        ACTOR_ID = os.getenv("ACTOR_ID")
        
        if not APIFY_TOKEN:
            raise SystemExit("Set APIFY_TOKEN env var")
        
        url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/run-sync-get-dataset-items"
        params = {"token": APIFY_TOKEN, "format": "json"}
        payload = {
            "profileUrls": [linkedin_url],
            "maxConcurrency": 1
        }
        
        try:
            r = requests.post(url, params=params, json=payload, timeout=300)
            r.raise_for_status()
            items = r.json()
            
            if not items:
                print("No data returned.")
                return None
            
            return items[0]  # Return first profile
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching profile: {e}")
            return None
    

    def extract_basic_info(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic profile information."""
        
        return {
            "firstName": profile_data.get("firstName", "").strip(),
            "lastName": profile_data.get("lastName", "").strip(),
            "fullName": profile_data.get("fullName", "").strip(),
            "headline": profile_data.get("headline", ""),
            "location": profile_data.get("addressWithCountry", ""),
            "about": profile_data.get("about", ""),
        }
    

    def extract_experience(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract work experience information."""
        
        extracted_experiences = []
        
        for exp in experiences:
            experience = {
                "title": exp.get("title", ""),
                "company": exp.get("subtitle", "").split("·")[0].strip() if exp.get("subtitle") else "",
                "duration": exp.get("caption", ""),
                "description": []
            }
            
            # Extract descriptions from subcomponents
            if exp.get("subComponents"):
                for sub in exp["subComponents"]:
                    if sub.get("description"):
                        for desc in sub["description"]:
                            if desc.get("type") == "textComponent":
                                experience["description"].append(desc.get("text", ""))
            
            extracted_experiences.append(experience)
        
        return extracted_experiences
    

    def extract_skills(self, skills: List[Dict[str, Any]]) -> List[str]:
        """Extract skills information."""

        extracted_skills = []

        for skill in skills:
            skill_title = skill.get("title", "")
            if skill_title:
                # Add context about where the skill was used
                context = []
                if skill.get("subComponents"):
                    for sub in skill["subComponents"]:
                        if sub.get("description"):
                            for desc in sub["description"]:
                                if desc.get("type") == "insightComponent":
                                    context.append(desc.get("text", ""))
                
                skill_info = {
                    "skill": skill_title,
                    "context": context
                }
                extracted_skills.append(skill_info)
        
        return extracted_skills
    

    def extract_education(self, educations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract education information."""
        
        extracted_education = []
        
        for edu in educations:
            education = {
                "institution": edu.get("title", ""),
                "degree": edu.get("subtitle", ""),
                "duration": edu.get("caption", ""),
                "description": []
            }
            
            # Extract descriptions
            if edu.get("subComponents"):
                for sub in edu["subComponents"]:
                    if sub.get("description"):
                        for desc in sub["description"]:
                            if desc.get("type") == "textComponent":
                                education["description"].append(desc.get("text", ""))
            
            extracted_education.append(education)
        
        return extracted_education
    

    def extract_projects(self, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract projects information."""
        
        extracted_projects = []
        
        for project in projects:
            project_info = {
                "title": project.get("title", ""),
                "duration": project.get("subtitle", ""),
                "description": []
            }
            
            # Extract descriptions
            if project.get("subComponents"):
                for sub in project["subComponents"]:
                    if sub.get("description"):
                        for desc in sub["description"]:
                            if desc.get("type") == "textComponent":
                                project_info["description"].append(desc.get("text", ""))
            
            extracted_projects.append(project_info)
        
        return extracted_projects
    

    def extract_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract recommendations information."""
        
        extracted_recommendations = []
        
        for rec_section in recommendations:
            section_name = rec_section.get("section_name", "")
            
            for component in rec_section.get("section_components", []):
                # Extract recommender info
                recommender_name = component.get("titleV2", "")
                recommender_title = component.get("subtitle", "")
                recommender_date = component.get("caption", "")
                
                # Extract recommendation text from subComponents
                recommendation_text = ""
                if component.get("subComponents"):
                    for sub in component["subComponents"]:
                        if sub.get("fixedListComponent"):
                            for item in sub["fixedListComponent"]:
                                if item.get("type") == "textComponent":
                                    recommendation_text = item.get("text", "")
                                    break
                
                # Only add if we have actual recommendation content
                if recommendation_text and section_name == "Received":
                    recommendation = {
                        "section": section_name,
                        "recommender_name": recommender_name,
                        "recommender_title": recommender_title,
                        "recommender_date": recommender_date,
                        "content": recommendation_text[:1000] + "..." if len(recommendation_text) > 1000 else recommendation_text
                    }
                    extracted_recommendations.append(recommendation)
        
        return extracted_recommendations
    

    def extract_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all relevant profile information into a clean dictionary."""
        
        extracted_profile = {
            "basic_info": self.extract_basic_info(profile_data),
            "experience": self.extract_experience(profile_data.get("experiences", [])),
            "skills": self.extract_skills(profile_data.get("skills", [])),
            "education": self.extract_education(profile_data.get("educations", [])),
            "projects": self.extract_projects(profile_data.get("projects", [])),
            "recommendations": self.extract_recommendations(profile_data.get("recommendations", []))
        }
        
        return extracted_profile
    

    def process_profile(self, linkedin_url: str) -> Optional[Dict[str, Any]]:
        """Process LinkedIn profile: fetch and extract relevant information."""
        
        raw_profile = self.fetch_linkedin_profile(linkedin_url)
        
        if not raw_profile:
            return None
        
        extracted_profile = self.extract_profile(raw_profile)
        
        return extracted_profile
    
    # Generating a manual summary of the profile
    def get_profile_summary(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of the extracted profile data."""
        
        if not profile_data:
            return {}
        
        summary = {
            "total_experience": len(profile_data.get("experience", [])),
            "total_skills": len(profile_data.get("skills", [])),
            "total_education": len(profile_data.get("education", [])),
            "total_projects": len(profile_data.get("projects", [])),
            "total_recommendations": len(profile_data.get("recommendations", [])),
            "has_about": bool(profile_data.get("basic_info", {}).get("about")),
            "has_location": bool(profile_data.get("basic_info", {}).get("location"))
        }
        
        return summary
    

def main():
    """Main function to extract the profile."""
    extractor = LinkedInProfileExtractor()
    
    linkedin_url = "https://www.linkedin.com/in/jay-goenka-b797851b2/"
    
    extracted_profile = extractor.process_profile(linkedin_url)
    
    if extracted_profile:
        print("Extracted Profile Structure:")
        print(json.dumps(extracted_profile, indent=2, ensure_ascii=False))
        print(f"\nProfile extracted successfully with {len(extracted_profile.get('experience', []))} experiences and {len(extracted_profile.get('skills', []))} skills")
        
        # Return the extracted profile for further use
        return extracted_profile
    else:
        print("Failed to extract profile")
        return None


if __name__ == "__main__":
    result = main()
    if result:
        print("Profile extraction completed successfully!")
    else:
        print("Profile extraction failed!")
    