#!/usr/bin/env python3
"""
Test script for the LinkedIn Profile Extractor
"""

import json
import sys
import os

# Add parent directory to path to import Profile_Extractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Profile_Extractor import LinkedInProfileExtractor

def test_sample_extraction():
    """Test the extractor with sample data."""
    extractor = LinkedInProfileExtractor()
    
    try:
        # Load sample data from test folder
        with open("Sample.txt", "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        
        print("Sample data loaded successfully!")
        print(f"Sample data keys: {list(sample_data.keys())}")
        
        # Extract relevant information
        extracted_profile = extractor.extract_profile(sample_data)
        
        print("\n" + "="*50)
        print("EXTRACTED PROFILE STRUCTURE:")
        print("="*50)
        
        # Print basic info
        print(f"Name: {extracted_profile['basic_info']['firstName']} {extracted_profile['basic_info']['lastName']}")
        print(f"Headline: {extracted_profile['basic_info']['headline']}")
        print(f"Location: {extracted_profile['basic_info']['location']}")
        
        # Print experience summary
        print(f"\nExperience Count: {len(extracted_profile['experience'])}")
        for i, exp in enumerate(extracted_profile['experience'][:2]):  # Show first 2
            print(f"  {i+1}. {exp['title']} at {exp['company']} ({exp['duration']})")
        
        # Print skills summary
        print(f"\nSkills Count: {len(extracted_profile['skills'])}")
        for i, skill in enumerate(extracted_profile['skills'][:5]):  # Show first 5
            print(f"  {i+1}. {skill['skill']}")
        
        # Print education summary
        print(f"\nEducation Count: {len(extracted_profile['education'])}")
        for edu in extracted_profile['education']:
            print(f"  - {edu['institution']}: {edu['degree']}")
        
        # Print projects summary
        print(f"\nProjects Count: {len(extracted_profile['projects'])}")
        for project in extracted_profile['projects']:
            print(f"  - {project['title']} ({project['duration']})")
        
        # Print recommendations summary
        print(f"\nRecommendations Count: {len(extracted_profile['recommendations'])}")
        for i, rec in enumerate(extracted_profile['recommendations']):
            print(f"  {i+1}. {rec['recommender_name']} ({rec['recommender_title']})")
            print(f"     Date: {rec['recommender_date']}")
            print(f"     Content: {rec['content'][:100]}...")
        
        # Save extracted profile
        with open("test_extracted_profile.json", "w", encoding="utf-8") as f:
            json.dump(extracted_profile, f, indent=2, ensure_ascii=False)
        
        print(f"\nExtracted profile saved to 'test_extracted_profile.json'")
        print(f"Total extracted data size: {len(json.dumps(extracted_profile))} characters")
        
        # Show LLM-ready format
        print("\n" + "="*50)
        print("LLM-READY FORMAT EXAMPLE:")
        print("="*50)
        
        llm_profile = {
            "candidate_name": f"{extracted_profile['basic_info']['firstName']} {extracted_profile['basic_info']['lastName']}",
            "headline": extracted_profile['basic_info']['headline'],
            "location": extracted_profile['basic_info']['location'],
            "summary": extracted_profile['basic_info']['about'][:300] + "..." if len(extracted_profile['basic_info']['about']) > 300 else extracted_profile['basic_info']['about'],
            "key_experiences": len(extracted_profile['experience']),
            "top_skills": len(extracted_profile['skills']),
            "education": len(extracted_profile['education']),
            "key_projects": len(extracted_profile['projects']),
            "recommendations_count": len(extracted_profile['recommendations'])
        }
        
        print(json.dumps(llm_profile, indent=2, ensure_ascii=False))
        print(f"\nLLM-ready format size: {len(json.dumps(llm_profile))} characters")
        
    except FileNotFoundError:
        print("Sample.txt not found in test folder!")
    except json.JSONDecodeError as e:
        print(f"Error parsing Sample.txt: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_sample_extraction()
