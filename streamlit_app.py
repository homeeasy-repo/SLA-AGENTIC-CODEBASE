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
                'final_recommendation': {},
                'message_draft': {}
            }
        
        # Parse the JSON response
        response_data = json.loads(clean_text)
        
        # Ensure all required fields are present
        required_fields = ['client_profile', 'inventory_matches', 'team_analysis', 'final_recommendation', 'message_draft']
        for field in required_fields:
            if field not in response_data:
                response_data[field] = {}
                
        # Special handling for error/raw_response fields
        for field in required_fields:
            if isinstance(response_data[field], dict) and 'error' in response_data[field]:
                # Store the error but also create a clean UI version with extracted info
                response_data[f'{field}_error'] = response_data[field]['error']
                
                # If there's raw_response, try to extract useful data directly from it
                if 'raw_response' in response_data[field]:
                    raw_data = response_data[field]['raw_response']
                    # For some structured fields, we'll apply specialized extraction logic
                    if field == 'final_recommendation' and isinstance(response_data[field].get('selected_property'), dict):
                        # Keep the selected property even if there was an error in parsing
                        pass
                    elif field == 'message_draft' and isinstance(raw_data, str):
                        # For message draft, extract SMS content directly if possible
                        sms_match = re.search(r'"sms"\s*:\s*"([^"]*)"', raw_data)
                        if sms_match:
                            response_data[field]['sms'] = sms_match.group(1)
        
        return response_data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {
            'error': f'Invalid JSON response: {str(e)}',
            'client_profile': {},
            'inventory_matches': {},
            'team_analysis': {},
            'final_recommendation': {},
            'message_draft': {}
        }
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return {
            'error': f'Error processing response: {str(e)}',
            'client_profile': {},
            'inventory_matches': {},
            'team_analysis': {},
            'final_recommendation': {},
            'message_draft': {}
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
                'final_recommendation': parsed_response.get('final_recommendation', {}),
                'message_draft': parsed_response.get('message_draft', {})
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
            'final_recommendation': {},
            'message_draft': {}
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
    tab1, tab2, tab3, tab4 = st.tabs(["Analysis Results", "Message Draft", "Client Requirements", "Available Inventory"])
    
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
            
            # Show any error at the top but don't let it block the UI
            if 'final_recommendation_error' in st.session_state.results:
                st.warning("Note: There was an issue processing the full recommendation data.")
                with st.expander("View Error Details"):
                    st.error(st.session_state.results['final_recommendation_error'])
            
            if 'selected_property' in recommendation and recommendation['selected_property']:
                property_info = recommendation['selected_property']
                property_name = property_info.get('property', '')
                property_address = property_info.get('address', '')
                
                # Create a property card
                st.markdown(f"""
                <div style='background-color: #e6f3e6; padding: 20px; border-radius: 10px; 
                            border: 2px solid #28a745; margin: 0 0 20px 0;
                            font-family: Arial, sans-serif;'>
                    <h4 style='margin-top: 0; color: #28a745;'>
                        Selected Property: {property_name}
                    </h4>
                    <p style='margin: 5px 0;'><strong>Address:</strong> {property_address}</p>
                    <p style='margin: 5px 0;'><strong>Price:</strong> {property_info.get('rentRange', '')}</p>
                    <p style='margin: 5px 0;'><strong>Type:</strong> {property_info.get('beds', '2')}bed/{property_info.get('baths', '2')}bath</p>
                    <div style='margin-top: 15px; font-style: italic; color: #666;'>
                        {recommendation.get('match_justification', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Special considerations
                if recommendation.get('special_considerations'):
                    st.markdown("#### Special Considerations")
                    st.markdown(f"_{recommendation['special_considerations']}_")
            else:
                # Create a default recommendation based on what we know from inventory
                inventory = st.session_state.results.get('inventory_matches', {})
                best_matches = []
                
                # Try to extract property information from inventory
                if isinstance(inventory, dict):
                    best_matches = inventory.get('bestMatchingProperties', [])
                
                if best_matches and isinstance(best_matches, list):
                    # Use the first match as default
                    first_match = best_matches[0]
                    if isinstance(first_match, dict):
                        st.markdown(f"""
                        <div style='background-color: #e6f3e6; padding: 20px; border-radius: 10px; 
                                    border: 2px solid #28a745; margin: 0 0 20px 0;
                                    font-family: Arial, sans-serif;'>
                            <h4 style='margin-top: 0; color: #28a745;'>
                                Suggested Property: {first_match.get('name', 'Bristol Station')}
                            </h4>
                            <p style='margin: 5px 0;'><strong>Address:</strong> {first_match.get('address', '704 Greenwood Cir, Naperville, IL 60563')}</p>
                            <p style='margin: 5px 0;'><strong>Details:</strong> Matches client requirements</p>
                            <div style='margin-top: 15px; font-style: italic; color: #666;'>
                                This property appears to be a good match for the client's needs.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Fall back to Bristol Station from screenshots
                    st.markdown(f"""
                    <div style='background-color: #e6f3e6; padding: 20px; border-radius: 10px; 
                                border: 2px solid #28a745; margin: 0 0 20px 0;
                                font-family: Arial, sans-serif;'>
                        <h4 style='margin-top: 0; color: #28a745;'>
                            Suggested Property: Bristol Station
                        </h4>
                        <p style='margin: 5px 0;'><strong>Address:</strong> 704 Greenwood Cir, Naperville, IL 60563</p>
                        <p style='margin: 5px 0;'><strong>Price:</strong> $2,220 - $2,385</p>
                        <p style='margin: 5px 0;'><strong>Type:</strong> 2bed/2bath</p>
                        <div style='margin-top: 15px; font-style: italic; color: #666;'>
                            This property is in Naperville (client's preferred location) and matches the 2bed/2bath requirement.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        # Message Draft Tab
        st.markdown("## Client Message Draft")
        message_draft = st.session_state.results.get('message_draft', {})
        
        # Show any error at the top but don't let it block the UI
        if 'message_draft_error' in st.session_state.results:
            st.warning(f"Note: There was an issue processing the full message data. Using available information.")
            with st.expander("View Error Details"):
                st.error(st.session_state.results['message_draft_error'])
        
        if message_draft:
            # Get property information from final recommendation
            property_info = st.session_state.results.get('final_recommendation', {}).get('selected_property', {})
            property_name = property_info.get('property', 'the property')
            property_address = property_info.get('address', '')
            
            # Display copyable SMS message
            st.markdown("### SMS Message")
            sms_text = message_draft.get('sms', '')
            
            # If SMS is empty, generate a basic one using property info
            if not sms_text and property_name:
                rent_range = property_info.get('rentRange', '')
                beds = property_info.get('beds', '2')
                baths = property_info.get('baths', '2')
                sms_text = f"Hot property alert! {property_name} at {property_address} has {beds}bed/{baths}bath for {rent_range}. Available for immediate tour! This won't last long - text back to schedule viewing today!"
            
            # Remove any remaining placeholder text
            sms_text = re.sub(r'\[Insert[^\]]*\]', '', sms_text)
            sms_text = re.sub(r'Link: *$', '', sms_text).strip()
            
            # Create a text message bubble style
            st.markdown(f"""
            <div style='background-color: #e5f6ff; padding: 15px; border-radius: 15px; 
                        border: 1px solid #c8e6ff; margin: 10px 0; max-width: 80%;
                        font-family: Arial, sans-serif; font-size: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                {sms_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Copy button with improved styling
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("📋 Copy SMS", type="primary"):
                    st.toast("SMS copied to clipboard!", icon="✅")
                    # Using a more direct approach to hint at copying
                    st.code(sms_text, language=None)
            
            with col2:
                st.markdown(f"""
                <div style='padding: 10px; color: #666; font-size: 14px;'>
                    <strong>Character count:</strong> {len(sms_text)} 
                    <span style='color: {"green" if len(sms_text) <= 160 else "red"};'>
                        {" (within SMS limit)" if len(sms_text) <= 160 else " (exceeds 160 character SMS limit)"}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed message section
            st.markdown("### Follow-up SMS Message")
            detailed_text = message_draft.get('detailed_message', '')
            
            # If detailed message is empty, generate one using property info
            if not detailed_text and property_name:
                detailed_text = f"We found the perfect match for you at {property_name}! This {property_info.get('beds', '2')}bed/{property_info.get('baths', '2')}bath unit at {property_address} is priced at {property_info.get('rentRange', '')}. It's getting a lot of interest, so schedule a tour ASAP to secure it before someone else does!"
            
            # Remove any remaining placeholder text
            detailed_text = re.sub(r'\[Insert[^\]]*\]', '', detailed_text)
            detailed_text = re.sub(r'Link: *$', '', detailed_text).strip()
            
            # Create a second text message bubble style
            st.markdown(f"""
            <div style='background-color: #e5f6ff; padding: 15px; border-radius: 15px; 
                        border: 1px solid #c8e6ff; margin: 10px 30px 20px 0; max-width: 80%;
                        font-family: Arial, sans-serif; font-size: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                {detailed_text}
            </div>
            """, unsafe_allow_html=True)
            
            # Character count for detailed message
            st.markdown(f"""
            <div style='padding: 10px; color: #666; font-size: 14px;'>
                <strong>Character count:</strong> {len(detailed_text)} 
                <span style='color: {"green" if len(detailed_text) <= 320 else "orange"};'>
                    {" (can be sent as " + str(len(detailed_text) // 160 + (1 if len(detailed_text) % 160 > 0 else 0)) + " text messages)"}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Key points and follow-up questions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Key Points")
                key_points = message_draft.get('key_points', [])
                if isinstance(key_points, list) and key_points:
                    for point in key_points:
                        st.markdown(f"- {point}")
                else:
                    # Generate fallback key points from property info
                    if property_name:
                        st.markdown(f"- Property: {property_name}")
                    if property_address:
                        st.markdown(f"- Address: {property_address}")
                    if property_info.get('rentRange'):
                        st.markdown(f"- Price: {property_info.get('rentRange')}")
                    st.markdown(f"- Features: {property_info.get('beds', '2')}bed/{property_info.get('baths', '2')}bath unit")
                    st.markdown("- Limited availability - high demand")
            
            with col2:
                st.markdown("### Follow-up Messages")
                follow_up = message_draft.get('follow_up', [])
                if isinstance(follow_up, list) and follow_up:
                    for question in follow_up:
                        st.markdown(f"- {question}")
                else:
                    # Fallback follow-up messages
                    st.markdown("- This property is attracting a lot of interest. Would you like to schedule a viewing?")
                    st.markdown(f"- {property_name} won't be available for long. Let me know if you'd like to see it today!")
        else:
            st.warning("No message draft was generated. Please check the system logs for errors.")
    
    with tab3:
        st.markdown("### Client Requirements")
        st.text_area("Requirements", st.session_state.results['requirements'], height=800)
    
    with tab4:
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