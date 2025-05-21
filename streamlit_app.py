import streamlit as st
import asyncio
import sys
from io import StringIO
from langcahin import process_client, get_client_markdown_content, get_client_requirements
import json
import re
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set page config
st.set_page_config(
    page_title="Property Matching System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = None

# Title and description
st.title("🏠 Property Matching System")
st.markdown("""
This system uses AI agents to match clients with suitable properties based on their requirements and preferences.
""")

def clean_ansi_codes(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_json_response(response_text):
    """Parse the JSON response from the agents."""
    try:
        # Clean any ANSI codes
        clean_text = clean_ansi_codes(response_text)
        
        # Ensure we have a non-empty string
        if not clean_text or clean_text.isspace():
            return {
                'error': 'Empty response received',
                'client_profile': {},
                'inventory_matches': {},
                'team_analysis': {},
                'final_recommendation': {}
            }
        
        # Parse the JSON response
        response_data = json.loads(clean_text)
        
        # Ensure all required fields are present
        required_fields = ['client_profile', 'inventory_matches', 'team_analysis', 'final_recommendation']
        for field in required_fields:
            if field not in response_data:
                response_data[field] = {}
        
        return response_data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {
            'error': f'Invalid JSON response: {str(e)}',
            'client_profile': {},
            'inventory_matches': {},
            'team_analysis': {},
            'final_recommendation': {}
        }
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return {
            'error': f'Error processing response: {str(e)}',
            'client_profile': {},
            'inventory_matches': {},
            'team_analysis': {},
            'final_recommendation': {}
        }

def clear_results():
    """Clear all results from session state."""
    st.session_state.results = None

async def process_client_data(client_id):
    """Process client data asynchronously."""
    try:
        # Get client requirements and inventory
        client_requirements = get_client_requirements(client_id)
        inventory_content = get_client_markdown_content(client_id)
        
        # Capture stdout
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Process the client using the new Langchain implementation
            response = await process_client(client_id)
            
            # Parse the response
            parsed_response = parse_json_response(response)
            
            # Check for errors in the response
            if 'error' in parsed_response:
                st.error(f"Error in response: {parsed_response['error']}")
            
            # Return results
            return {
                'output': captured_output.getvalue(),
                'requirements': client_requirements,
                'inventory': inventory_content,
                'client_profile': parsed_response.get('client_profile', {}),
                'inventory_matches': parsed_response.get('inventory_matches', {}),
                'team_analysis': parsed_response.get('team_analysis', {}),
                'final_recommendation': parsed_response.get('final_recommendation', {})
            }
        finally:
            # Restore stdout
            sys.stdout = old_stdout
    except Exception as e:
        print(f"Error in process_client_data: {str(e)}")
        error_response = {
            'error': str(e),
            'client_profile': {},
            'inventory_matches': {},
            'team_analysis': {},
            'final_recommendation': {}
        }
        raise Exception(f"Error processing client data: {str(e)}")

# Sidebar for input
with st.sidebar:
    st.header("Input Parameters")
    client_id = st.number_input("Client ID", min_value=1, value=691481)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Client", type="primary"):
            if client_id:
                try:
                    # Clear previous results
                    clear_results()
                    
                    # Create a progress indicator
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process the client
                    status_text.text("Processing client data...")
                    progress_bar.progress(25)
                    
                    # Run the async function
                    results = asyncio.run(process_client_data(client_id))
                    
                    # Update progress
                    progress_bar.progress(75)
                    status_text.text("Updating display...")
                    
                    # Store results in session state
                    st.session_state.results = results
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    # Clear progress indicators after a delay
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing client: {str(e)}")
            else:
                st.warning("Please enter a Client ID")
    
    with col2:
        if st.button("Clear Results"):
            clear_results()
            st.rerun()

# Main content area
if st.session_state.results:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Analysis Results", "Client Requirements", "Available Inventory"])
    
    with tab1:
        # Create three columns for the main sections
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Client Profile Analysis
            st.markdown("### Client Profile Analysis")
            profile_data = st.session_state.results['client_profile']
            st.json(profile_data)
            
            # Inventory Matches
            st.markdown("### Inventory Matches")
            matches_data = st.session_state.results['inventory_matches']
            st.json(matches_data)
        
        with col2:
            # Team Analysis
            st.markdown("### Team Analysis")
            team_data = st.session_state.results['team_analysis']
            st.json(team_data)
        
        with col3:
            # Final Recommendation
            st.markdown("### Final Recommendation")
            recommendation = st.session_state.results['final_recommendation']
            if recommendation:
                st.markdown(f"""
                <div style='background-color: #e6f3e6; padding: 1rem; border-radius: 0.5rem; border: 2px solid #28a745; margin: 1rem 0;'>
                    <pre style='margin: 0; white-space: pre-wrap;'>{json.dumps(recommendation, indent=2)}</pre>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No recommendation generated")
    
    with tab2:
        st.markdown("### Client Requirements")
        st.text_area("Requirements", st.session_state.results['requirements'], height=800)
    
    with tab3:
        st.markdown("### Available Inventory")
        st.text_area("Inventory", st.session_state.results['inventory'], height=800)

# Add some styling
st.markdown("""
<style>
    .stJson {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .stMarkdown h3 {
        color: #28a745;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        background-color: #28a745;
        color: white;
    }
    .stButton button:hover {
        background-color: #218838;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #28a745;
        color: white;
    }
    pre {
        font-family: monospace;
        font-size: 14px;
        line-height: 1.5;
        color: #333;
    }
</style>
""", unsafe_allow_html=True) 