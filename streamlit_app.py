import streamlit as st
import json
import tempfile
from profile_extractor import LinkedInProfileExtractor
from profile_processor import ComprehensiveProfileProcessor

def main():
    
    st.set_page_config(
        page_title="Candidate Analyzer",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("Candidate Analyzer")
    st.markdown("Analyze candidates using LinkedIn profiles and/or resumes")
    
    # Initialize components
    try:
        extractor = LinkedInProfileExtractor()
        processor = ComprehensiveProfileProcessor()
        st.success("Components initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.stop()
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resume Upload")
        uploaded_file = st.file_uploader(
            "Upload resume",
            type=['pdf', 'docx', 'txt'],
            help="Optional: Upload candidate's resume"
        )
        
        resume_data = None
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            st.success(f"{uploaded_file.name} uploaded")
            
            # Process resume immediately
            with st.spinner("Processing resume..."):
                resume_result = processor.process_resume_file(tmp_file_path)
                if resume_result:
                    resume_data = resume_result.get('data')
                    parser_used = resume_result.get('parser_used', 'unknown')
                    st.info(f"Parsed using: {parser_used}")
                else:
                    st.warning("Resume processing failed")

    with col2:
        st.subheader("LinkedIn Profile")
        linkedin_url = st.text_input(
            "LinkedIn URL",
            placeholder="https://www.linkedin.com/in/username/",
            help="Optional: Enter LinkedIn profile URL"
        )
        
        linkedin_data = None
        if linkedin_url:
            if extractor.validate_linkedin_url(linkedin_url):
                st.success("Valid LinkedIn URL")
                
                if st.button("🔍 Extract Profile", type="primary"):
                    with st.spinner("Extracting LinkedIn data..."):
                        linkedin_data = processor.process_linkedin_profile(linkedin_url)
                        if linkedin_data:
                            st.success("LinkedIn data extracted")
                        else:
                            st.error("LinkedIn extraction failed")
            else:
                st.warning("Invalid LinkedIn URL format")

    # Generate report section
    if linkedin_data or resume_data:
        st.markdown("---")
        st.subheader("Candidate Report")

        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                report = processor.generate_report(linkedin_data, resume_data)
                
                if report:
                    # Display report
                    st.markdown("### Comprehensive Analysis")
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="candidate_report.txt",
                        mime="text/plain"
                    )
                    
                    # Quick stats
                    if linkedin_data:
                        st.markdown("### Quick Stats")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            exp_count = len(linkedin_data.get('experience', []))
                            st.metric("Experience", f"{exp_count} positions")
                        
                        with col2:
                            skills_count = len(linkedin_data.get('skills', []))
                            st.metric("Skills", f"{skills_count} listed")
                        
                        with col3:
                            edu_count = len(linkedin_data.get('education', []))
                            st.metric("Education", f"{edu_count} institutions")
                        
                        with col4:
                            has_resume = "Yes" if resume_data else "No"
                            st.metric("Resume", has_resume)
                
                else:
                    st.error("Report generation failed")

    # Sidebar
    with st.sidebar:
        st.markdown("### How to Use")
        st.markdown("""
        1. **Upload resume** (PDF, DOCX, TXT)
        2. **Enter LinkedIn URL** (optional)
        3. **Click Extract Profile** if using LinkedIn
        4. **Generate Report** for analysis
        
        **Note:** You can use either LinkedIn URL, resume, or both for comprehensive analysis.
        """)
        
        if 'APIFY_TOKEN' not in st.secrets:
            st.warning("Set APIFY_TOKEN for LinkedIn extraction")

        # Debug info
        if st.checkbox("Debug Info"):
            st.json({
                "LinkedIn Data": bool(linkedin_data),
                "Resume Data": bool(resume_data),
                "LLM Available": processor.llm_available
            })

if __name__ == "__main__":
    main()
