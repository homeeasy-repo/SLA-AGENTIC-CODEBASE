import streamlit as st
import asyncio
import json
from agno_multimodel import process_client, agent_team

st.set_page_config(
    page_title="Property Matching System",
    page_icon="🏠",
    layout="wide"
)

st.title("Property Matching System")
st.subheader("Enter Client ID to analyze and get property recommendations")

# Input for client ID
client_id = st.text_input("Enter Client ID", "")

if st.button("Analyze Client"):
    if client_id:
        try:
            # Create placeholder for responses
            client_profiler_box = st.empty()
            inventory_matcher_box = st.empty()
            team_coordinator_box = st.empty()
            final_response_box = st.empty()

            # Process the client and capture responses
            async def process_and_capture():
                # Store the original print function
                original_print = print
                responses = []

                # Custom print function to capture responses
                def custom_print(*args, **kwargs):
                    response = " ".join(str(arg) for arg in args)
                    responses.append(response)
                    original_print(*args, **kwargs)

                # Replace print with custom function
                import builtins
                builtins.print = custom_print

                try:
                    # Process the client
                    await process_client(int(client_id))
                    
                    # Format responses
                    client_profiler_response = ""
                    inventory_matcher_response = ""
                    team_coordinator_response = ""
                    final_message = ""

                    # Combine all responses into a single string
                    full_response = "\n".join(responses)
                    
                    # Extract responses using string markers
                    def extract_response(text, start_marker, end_marker):
                        if start_marker in text:
                            start_idx = text.find(start_marker)
                            end_idx = text.find(end_marker, start_idx)
                            if start_idx >= 0 and end_idx >= 0:
                                section = text[start_idx:end_idx]
                                # Remove box markers and clean up
                                section = section.replace(start_marker, "")
                                section = section.replace("┃", "").strip()
                                return section
                        return ""

                    # Extract each response
                    client_profiler_response = extract_response(full_response, "┏━ Client Profiler Response ━━", "┗━")
                    inventory_matcher_response = extract_response(full_response, "┏━ Inventory Matcher Response ━━", "┗━")
                    team_coordinator_response = extract_response(full_response, "┏━ Team Coordinator Response ━━", "┗━")
                    
                    # Extract final message
                    response_section = extract_response(full_response, "┏━ Response ━━", "┗━")
                    if "Hi" in response_section:
                        final_message = response_section[response_section.find("Hi"):].strip()

                    # Display responses in boxes
                    with client_profiler_box.container():
                        st.subheader("Client Profiler Analysis")
                        st.text_area("Analysis", client_profiler_response, height=300, key="client_profiler_text")

                    with inventory_matcher_box.container():
                        st.subheader("Inventory Matcher Analysis")
                        st.text_area("Analysis", inventory_matcher_response, height=300, key="inventory_matcher_text")

                    with team_coordinator_box.container():
                        st.subheader("Team Coordinator Analysis")
                        st.text_area("Analysis", team_coordinator_response, height=300, key="team_coordinator_text")

                    # Display final response
                    with final_response_box.container():
                        st.subheader("Final Property Message")
                        if final_message:
                            st.success(final_message)
                            # Also show the raw JSON
                            st.json({"message": final_message})
                        else:
                            st.warning("No property message generated")

                finally:
                    # Restore original print function
                    builtins.print = original_print

            # Run the async function
            asyncio.run(process_and_capture())

        except Exception as e:
            st.error(f"Error processing client: {str(e)}")
    else:
        st.warning("Please enter a Client ID")

# Add some styling
st.markdown("""
<style>
    .stTextArea textarea {
        font-family: monospace;
    }
    .stJson {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stSuccess {
        background-color: #e6f3e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
    }
</style>
""", unsafe_allow_html=True) 