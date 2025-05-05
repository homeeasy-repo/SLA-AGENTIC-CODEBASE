import streamlit as st
import os
from inventory_agent import get_client_requirements, get_client_markdown_content, analyze_properties
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Inventory Analysis",
    page_icon="👑",
    layout="wide"
)

# Add title and description
st.title("Real Estate Inventory Analysis")
st.markdown("Enter a client ID to analyze property listings and find the best match.")

# Create input section
col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    # Client ID input
    client_id = st.number_input("Enter Client ID", min_value=1, step=1)
    
    # Create a row for buttons
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        # Add analyze button
        analyze_button = st.button("Analyze Properties")
    
    with button_col2:
        # Add clear button
        clear_button = st.button("Clear Results")

# Initialize session state for results if not exists
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'markdown_content' not in st.session_state:
    st.session_state.markdown_content = None
if 'client_requirements' not in st.session_state:
    st.session_state.client_requirements = None

# Clear all results when clear button is clicked
if clear_button:
    st.session_state.analysis_results = None
    st.session_state.markdown_content = None
    st.session_state.client_requirements = None
    st.rerun()

# When analyze button is clicked
if analyze_button and client_id:
    try:
        # Get client requirements
        client_requirements = get_client_requirements(client_id)
        st.session_state.client_requirements = client_requirements
        
        # Get markdown content
        markdown_content = get_client_markdown_content(client_id)
        st.session_state.markdown_content = markdown_content
        
        # Analyze properties
        results = analyze_properties(client_id, markdown_content)
        st.session_state.analysis_results = results
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Create three columns for display
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Client Requirements")
    if st.session_state.client_requirements:
        st.text_area(
            "Requirements",
            st.session_state.client_requirements,
            height=400,
            disabled=True,
            help="Client's specified requirements and preferences"
        )
    else:
        st.info("Client requirements will appear here.")

with col2:
    st.subheader("Inventory Markdown")
    if st.session_state.markdown_content:
        st.text_area(
            "Property Listings",
            st.session_state.markdown_content,
            height=400,
            disabled=True,
            help="Available property listings"
        )
    else:
        st.info("Enter a client ID and click 'Analyze Properties' to view inventory.")

with col3:
    st.subheader("Analysis Results")
    if st.session_state.analysis_results:
        st.markdown(st.session_state.analysis_results)
    else:
        st.info("Analysis results will appear here after processing.")

# Add footer with timestamp
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    # Add a download button for the analysis results
    if st.session_state.analysis_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"property_analysis_{client_id}_{timestamp}.txt"
        st.download_button(
            label="Download Analysis",
            data=st.session_state.analysis_results,
            file_name=filename,
            mime="text/plain"
        ) 