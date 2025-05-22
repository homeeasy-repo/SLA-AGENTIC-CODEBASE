import streamlit as st
import asyncio
import sys
import re
import json
from io import StringIO
from langcahin import process_client
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set page config
st.set_page_config(
    page_title="🏠 AI Property Matching System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'agent_thinking' not in st.session_state:
    st.session_state.agent_thinking = ""

def extract_client_id_from_url(url: str) -> int:
    """Extract client ID from Follow Up Boss URL."""
    try:
        # Remove any whitespace
        url = url.strip()
        
        # Extract the last digits from the URL
        match = re.search(r'/(\d+)/?$', url)
        if match:
            return int(match.group(1))
        
        # If no match, try to find any digits at the end
        digits = re.findall(r'\d+', url)
        if digits:
            return int(digits[-1])
        
        return None
    except:
        return None

def clear_results():
    """Clear all results from session state."""
    st.session_state.results = None
    st.session_state.agent_thinking = ""

async def process_client_with_thinking(client_id: int):
    """Process client and capture the agent's thinking process."""
    try:
        # Capture stdout to show agent thinking
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Process the client using the new Main Property Agent
            result_json = await process_client(client_id)
            
            # Get the captured thinking process
            thinking_output = captured_output.getvalue()
            
            # Parse the final result
            result = json.loads(result_json)
            
            return {
                'thinking': thinking_output,
                'final_result': result,
                'success': True
            }
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
    except Exception as e:
        return {
            'thinking': f"Error processing client: {str(e)}",
            'final_result': {
                'summary': f"Error processing client {client_id}: {str(e)}",
                'message': "Hi! Let me look into some options for you and get back to you soon.",
                'inventory_found': False
            },
            'success': False
        }

# Title and description
st.title("🏠 AI Property Matching System")
st.markdown("""
**Main Property Agent** - Single agent that coordinates everything internally!
- 🧠 **Smart Decision Making**: Agent decides which tools to call
- 🔄 **Live Thinking**: See the agent's reasoning process
- 📱 **Human Messages**: Natural SMS ready to send
""")

# Sidebar for input
with st.sidebar:
    st.header("Follow Up Boss Integration")
    
    # URL input
    url_input = st.text_area(
        "Paste Follow Up Boss URL:",
        placeholder="https://services.followupboss.com/2/people/view/706732",
        height=100
    )
    
    # Extract client ID
    client_id = None
    if url_input:
        client_id = extract_client_id_from_url(url_input)
        if client_id:
            st.success(f"✅ Client ID extracted: {client_id}")
        else:
            st.error("❌ Could not extract client ID from URL")
    
    # Manual client ID input (backup)
    st.markdown("---")
    st.subheader("Or Enter Client ID Directly")
    manual_client_id = st.number_input("Client ID", min_value=1, value=691481 if not client_id else client_id)
    
    # Use manual ID if URL doesn't work
    if not client_id:
        client_id = manual_client_id
    
    st.markdown("---")
    
    # Process button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Process Client", type="primary", disabled=st.session_state.processing):
            if client_id:
                st.session_state.processing = True
                st.rerun()
            else:
                st.warning("Please provide a valid client ID")
    
    with col2:
        if st.button("🗑️ Clear Results"):
            clear_results()
            st.rerun()

# Main processing logic
if st.session_state.processing and client_id:
    # Clear previous results
    clear_results()
    
    # Create columns for live updates
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🧠 Agent Thinking Process")
        thinking_container = st.container()
        
        with thinking_container:
            # Show processing message
            progress_bar = st.progress(0)
            status_text = st.empty()
            thinking_display = st.empty()
            
            status_text.text("🤖 Starting Main Property Agent...")
            progress_bar.progress(25)
            
            # Process client asynchronously
            with st.spinner("Agent is thinking and using tools..."):
                results = asyncio.run(process_client_with_thinking(client_id))
            
            # Update progress
            progress_bar.progress(75)
            status_text.text("✅ Processing complete!")
            
            # Store results
            st.session_state.results = results
            st.session_state.processing = False
            
            progress_bar.progress(100)
            status_text.text("🎉 Agent finished successfully!")
            
            # Auto-rerun to show results
            st.rerun()

# Display results if available
if st.session_state.results and not st.session_state.processing:
    results = st.session_state.results
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🧠 Agent Thinking", "📱 Final Output", "📊 JSON Response"])
    
    with tab1:
        st.markdown("### 🤖 Main Property Agent - Decision Making Process")
        
        if results.get('thinking'):
            # Display the thinking process with syntax highlighting
            thinking_text = results['thinking']
            
            # Clean up the thinking text for better display
            # Remove excessive whitespace but keep structure
            lines = thinking_text.split('\n')
            cleaned_lines = []
            for line in lines:
                if line.strip():  # Keep non-empty lines
                    cleaned_lines.append(line)
                elif cleaned_lines and cleaned_lines[-1].strip():  # Add single empty line between sections
                    cleaned_lines.append('')
            
            thinking_display = '\n'.join(cleaned_lines)
            
            # Show thinking in a code block with scrolling
            st.code(thinking_display, language=None)
            
            # Show key decision points
            if "Action:" in thinking_display:
                st.markdown("#### 🔧 Tools Used by Agent:")
                actions = re.findall(r'Action: (\w+)', thinking_display)
                for i, action in enumerate(actions, 1):
                    if action == "analyze_client_profile":
                        st.success(f"{i}. 👤 Analyzed Client Profile")
                    elif action == "find_inventory_match":
                        st.success(f"{i}. 🏢 Found Inventory Matches")
                    elif action == "generate_client_message":
                        st.success(f"{i}. 💬 Generated Human Message")
                    else:
                        st.info(f"{i}. 🛠️ {action}")
        else:
            st.warning("No thinking process captured.")
    
    with tab2:
        st.markdown("### 📱 Ready-to-Send SMS Message")
        
        final_result = results.get('final_result', {})
        
        # Display client summary
        st.markdown("#### 👤 Client Summary")
        summary = final_result.get('summary', 'No summary available')
        st.info(summary)
        
        # Display inventory match status
        inventory_found = final_result.get('inventory_found', False)
        if inventory_found:
            st.success("✅ **Property Match Found!**")
        else:
            st.warning("❌ **No Perfect Match Found**")
        
        # Display the message prominently
        st.markdown("#### 📱 SMS Message for Client")
        message = final_result.get('message', 'No message generated')
        
        # Create a phone-like message bubble
        st.markdown(f"""
        <div style='background-color: #007AFF; color: white; padding: 15px 20px; 
                    border-radius: 20px; max-width: 80%; margin: 20px 0; 
                    font-family: Arial, sans-serif; font-size: 16px; 
                    box-shadow: 0 2px 10px rgba(0,122,255,0.3);'>
            {message}
        </div>
        """, unsafe_allow_html=True)
        
        # Message statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Character Count", len(message))
        with col2:
            sms_ready = "✅ Yes" if len(message) <= 160 else "❌ Too Long"
            st.metric("SMS Ready", sms_ready)
        with col3:
            has_placeholders = '[' in message or 'insert' in message.lower()
            human_like = "✅ Yes" if not has_placeholders else "❌ Has Placeholders"
            st.metric("Human-like", human_like)
        
        # Copy button
        if st.button("📋 Copy Message to Clipboard", type="secondary"):
            st.success("Message ready to copy!")
            st.code(message, language=None)
    
    with tab3:
        st.markdown("### 📊 Complete JSON Response")
        
        final_result = results.get('final_result', {})
        
        # Display formatted JSON
        st.json(final_result)
        
        # Download option
        json_str = json.dumps(final_result, indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"client_{client_id}_analysis.json",
            mime="application/json"
        )

# Show example if no results
if not st.session_state.results and not st.session_state.processing:
    st.markdown("---")
    st.markdown("### 📝 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1:** Paste Follow Up Boss URL
        ```
        https://services.followupboss.com/2/people/view/706732
        ```
        
        **Step 2:** Click "Process Client"
        
        **Step 3:** Watch the agent think and decide which tools to use
        
        **Step 4:** Get your human-like SMS message!
        """)
    
    with col2:
        st.markdown("""
        **What You'll See:**
        - 🧠 Agent's decision-making process
        - 🔧 Which tools the agent chooses to use
        - 👤 Client profile analysis
        - 🏢 Property matching results
        - 📱 Ready-to-send SMS message
        - ✅ Simple True/False for inventory found
        """)

# Add some styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007AFF;
        color: white;
    }
    .stProgress .stProgress-bar {
        background-color: #007AFF;
    }
</style>
""", unsafe_allow_html=True) 