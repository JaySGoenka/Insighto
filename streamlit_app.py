"""
LinkedIn Profile Extractor Streamlit App

This Streamlit application provides a user interface for:
1. Uploading resumes
2. Entering LinkedIn URLs
3. Processing LinkedIn profiles using the Profile_Extractor module

"""

import streamlit as st
import pandas as pd
import json
from Profile_Extractor import LinkedInProfileExtractor
from profile_processor import ComprehensiveProfileProcessor
from smart_resume_parser import SmartResumeParser
import tempfile
import os

def main():
    st.set_page_config(
        page_title="LinkedIn Profile Extractor",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 LinkedIn Profile Extractor")
    st.markdown("Upload your resume and extract information from LinkedIn profiles")
    
    # Initialize the profile extractor, processor, and resume parser
    try:
        extractor = LinkedInProfileExtractor()
        processor = ComprehensiveProfileProcessor()
        resume_parser = SmartResumeParser()
        st.success("Profile Extractor, Processor, and Resume Parser initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()
    
    # Create two columns for the main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Resume Upload")
        
        # File uploader for resume
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=['pdf', 'docx', 'txt', 'doc'],
            help="Supported formats: PDF, DOCX, TXT, DOC"
        )
        
        if uploaded_file is not None:
            # Display file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            
            # Save file temporarily (for future processing)
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            st.success(f"Resume uploaded successfully: {uploaded_file.name}")
            
            # Parse and preview resume content
            st.subheader("📋 Resume Preview")
            try:
                parsed_resume = resume_parser.parse_resume(tmp_file_path)
                if "error" not in parsed_resume:
                    # Display parsing summary
                    parsing_summary = resume_parser.get_parsing_summary(parsed_resume)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sections Found", parsing_summary.get('total_sections', 0))
                    with col2:
                        st.metric("Parsing Quality", parsing_summary.get('parsing_quality', 'Unknown').title())
                    with col3:
                        st.metric("Basic Info", "Complete" if parsing_summary.get('basic_info_complete') else "Incomplete")
                    
                    # Show sections found
                    if parsing_summary.get('sections_found'):
                        st.write(f"**Sections detected:** {', '.join(parsing_summary['sections_found'])}")
                    
                    # Store the file path and parsed data in session state
                    st.session_state['resume_path'] = tmp_file_path
                    st.session_state['parsed_resume'] = parsed_resume
                    
                    st.success("✅ Resume parsed successfully!")
                else:
                    st.error(f"Failed to parse resume: {parsed_resume['error']}")
                    st.session_state['resume_path'] = tmp_file_path
            except Exception as e:
                st.error(f"Error parsing resume: {str(e)}")
                st.session_state['resume_path'] = tmp_file_path
    
    with col2:
        st.header("🔗 LinkedIn Profile")
        
        # Text input for LinkedIn URL
        linkedin_url = st.text_input(
            "Enter LinkedIn Profile URL",
            placeholder="https://www.linkedin.com/in/username/",
            help="Paste the full LinkedIn profile URL here"
        )
        
        # URL validation using the extractor's validation method
        if linkedin_url:
            if extractor.validate_linkedin_url(linkedin_url):
                st.success("Valid LinkedIn URL format")
            else:
                st.warning("Please enter a valid LinkedIn profile URL")
        
        # Process button
        process_button = st.button(
            "Process Profile",
            type="primary",
            disabled=not linkedin_url or not extractor.validate_linkedin_url(linkedin_url)
        )
    
    # Process the LinkedIn profile when button is clicked
    if process_button and linkedin_url:
        st.header("🔄 Processing Profile")
        
        with st.spinner("Extracting LinkedIn profile data..."):
            try:
                # Process the profile using the extractor
                extracted_profile = extractor.process_profile(linkedin_url)
                
                if extracted_profile:
                    st.success("Profile extracted successfully!")
                    
                    # Display the extracted profile in an expandable section
                    with st.expander("Extracted Profile Data", expanded=True):
                        st.json(extracted_profile)
                    
                    # Create a download button for the extracted data
                    profile_json = json.dumps(extracted_profile, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Download Profile Data (JSON)",
                        data=profile_json,
                        file_name="linkedin_profile_data.json",
                        mime="application/json"
                    )
                    
                    # Display summary statistics
                    st.subheader("Profile Summary")
                    
                    # Basic info summary
                    if extracted_profile.get('basic_info'):
                        basic = extracted_profile['basic_info']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Name", f"{basic.get('firstName', '')} {basic.get('lastName', '')}")
                        with col2:
                            st.metric("Location", basic.get('location', 'N/A'))
                        with col3:
                            st.metric("Headline", basic.get('headline', 'N/A')[:50] + "..." if len(basic.get('headline', '')) > 50 else basic.get('headline', 'N/A'))
                    
                    # Experience summary
                    if extracted_profile.get('experience'):
                        st.write(f"**Work Experience:** {len(extracted_profile['experience'])} positions")
                        
                        # Create a DataFrame for experience
                        exp_data = []
                        for exp in extracted_profile['experience']:
                            exp_data.append({
                                'Title': exp.get('title', 'N/A'),
                                'Company': exp.get('company', 'N/A'),
                                'Duration': exp.get('duration', 'N/A')
                            })
                        
                        if exp_data:
                            exp_df = pd.DataFrame(exp_data)
                            st.dataframe(exp_df, use_container_width=True)
                    
                    # Skills summary
                    if extracted_profile.get('skills'):
                        st.write(f"**Skills:** {len(extracted_profile['skills'])} skills identified")
                        
                        # Display skills in a nice format
                        skills_text = ", ".join([skill.get('skill', '') for skill in extracted_profile['skills']])
                        st.write(f"*{skills_text}*")
                    
                    # Education summary
                    if extracted_profile.get('education'):
                        st.write(f"**Education:** {len(extracted_profile['education'])} institutions")
                        
                        # Create a DataFrame for education
                        edu_data = []
                        for edu in extracted_profile['education']:
                            edu_data.append({
                                'Institution': edu.get('institution', 'N/A'),
                                'Degree': edu.get('degree', 'N/A'),
                                'Duration': edu.get('duration', 'N/A')
                            })
                        
                        if edu_data:
                            edu_df = pd.DataFrame(edu_data)
                            st.dataframe(edu_df, use_container_width=True)
                    
                    # Get and display profile summary
                    profile_summary = extractor.get_profile_summary(extracted_profile)
                    
                    # Display summary metrics
                    st.subheader("📊 Profile Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Experience", profile_summary.get('total_experience', 0))
                    with col2:
                        st.metric("Skills", profile_summary.get('total_skills', 0))
                    with col3:
                        st.metric("Education", profile_summary.get('total_education', 0))
                    with col4:
                        st.metric("Projects", profile_summary.get('total_projects', 0))
                    
                    # Store the extracted profile in session state for potential future use
                    st.session_state['extracted_profile'] = extracted_profile
                    
                    # Generate comprehensive report if resume is also available
                    if 'resume_path' in st.session_state:
                        st.subheader("📊 Comprehensive Candidate Report")
                        st.info("Generating comprehensive report using both LinkedIn and resume data...")
                        
                        with st.spinner("Processing resume and generating comprehensive report..."):
                            try:
                                # Process resume
                                resume_data = processor.process_resume_file(st.session_state['resume_path'])
                                
                                if resume_data and 'error' not in resume_data:
                                    # Generate comprehensive report
                                    comprehensive_report = processor.generate_comprehensive_report(
                                        extracted_profile, resume_data
                                    )
                                    
                                    if comprehensive_report:
                                        # Display the report in an expandable section
                                        with st.expander("📋 Comprehensive Candidate Report", expanded=True):
                                            st.markdown(comprehensive_report)
                                        
                                        # Download button for the comprehensive report
                                        st.download_button(
                                            label="📥 Download Comprehensive Report (TXT)",
                                            data=comprehensive_report,
                                            file_name="comprehensive_candidate_report.txt",
                                            mime="text/plain"
                                        )
                                        
                                        # Display candidate summary
                                        candidate_summary = processor.get_candidate_summary(extracted_profile, resume_data)
                                        st.subheader("📈 Candidate Summary")
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Name", candidate_summary.get('name', 'N/A'))
                                        with col2:
                                            st.metric("Experience", candidate_summary.get('total_experience', 0))
                                        with col3:
                                            st.metric("Skills", candidate_summary.get('total_skills', 0))
                                        with col4:
                                            st.metric("LLM Available", "Yes" if candidate_summary.get('llm_available') else "No")
                                        
                                        st.success("✅ Comprehensive report generated successfully!")
                                    else:
                                        st.error("Failed to generate comprehensive report")
                                else:
                                    st.warning("Resume processing failed. Showing LinkedIn profile only.")
                                    
                            except Exception as e:
                                st.error(f"Error generating comprehensive report: {str(e)}")
                                st.info("Showing LinkedIn profile data only.")
                    
                else:
                    st.error("Failed to extract profile. Please check the URL and try again.")
                    
            except Exception as e:
                st.error(f"Error processing profile: {str(e)}")
                st.error("Please check your API configuration and try again.")
    
    # Sidebar for additional information
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        **How to use:**
        1. Upload your resume (optional but recommended)
        2. Enter a LinkedIn profile URL
        3. Click 'Process Profile' to extract data
        4. Get comprehensive candidate report
        
        **Features:**
        - Resume upload and parsing
        - LinkedIn profile extraction
        - AI-powered comprehensive reports
        - Candidate summary and metrics
        - Download reports in multiple formats
        """)
        
        st.header("🔧 Configuration")
        st.info("Make sure you have set up your APIFY_TOKEN environment variable for LinkedIn data extraction.")
        
        # Display current session state info
        if 'resume_path' in st.session_state:
            st.success("✅ Resume uploaded")
        if 'extracted_profile' in st.session_state:
            st.success("✅ Profile extracted")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and LinkedIn Profile Extractor*")

if __name__ == "__main__":
    main()
