import streamlit as st
from agno_agents import main_agent  # Make sure this is correct

# FIRST STREAMLIT COMMAND
st.set_page_config(page_title="HomeEasy Sales Agent", page_icon="🏡", layout="wide")

# Now you can use session_state and Streamlit commands
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "inventory" not in st.session_state:
    st.session_state.inventory = "not available"
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "tools_used" not in st.session_state:
    st.session_state.tools_used = []

# Function to save chat history into a TXT file
def save_chat_to_txt():
    """Save the chat history to a TXT file."""
    with open("chat_history.txt", "w", encoding="utf-8") as f:
        for sender, message in st.session_state.chat_history:
            if sender == "client":
                f.write(f"Client: {message}\n")
            else:
                f.write(f"Agent: {message}\n")

# Initialize Main Agent
@st.cache_resource
def load_agent():
    return main_agent

main_agent = load_agent()

# Layout Columns
left_col, center_col, right_col = st.columns([1, 2, 1])

# LEFT COLUMN - NOTES SECTION
with left_col:
    st.header("📝 Notes")
    st.session_state.notes = st.text_area(
        label="Write your observations or mistakes here",
        value=st.session_state.notes,
        height=400
    )

# CENTER COLUMN - CHAT INTERFACE
with center_col:
    st.title("🏡 HomeEasy Sales Bot")
    st.caption("Talk naturally like a client. Agent will respond smartly with SMS drafts!")

    # Show Chat History
    for i, (sender, message) in enumerate(st.session_state.chat_history):
        if sender == "client":
            st.chat_message("user").markdown(message)
        else:
            st.chat_message("assistant").markdown(message)

    # Input Box for Client Message
    client_input = st.chat_input("Type your message as the client...")

    if client_input:
        # Add client message to history
        st.session_state.chat_history.append(("client", client_input))
        save_chat_to_txt()  # SAVE AFTER CLIENT MESSAGE

        # Prepare context for MainAgent
        # full_context = {
        #     "chat_history": client_input,
        #     "inventory_list": st.session_state.inventory
        # }
        full_chat_history = "\n".join(
             [f"{'Client' if sender == 'client' else 'Agent'}: {message}" for sender, message in st.session_state.chat_history]
        )

        full_context = {
            "chat_history": full_chat_history,
            "inventory_list": st.session_state.inventory
        }


        # Call MainAgent
        sms_response = main_agent.process_query({
            "chat_history": full_chat_history,
            "inventory_list": st.session_state.inventory
            
        })

        # Log the tools used (manual simulation since real tools called inside MainAgent hidden)
        st.session_state.tools_used.append("MainAgent + SMSFormatterAgent")

        # Add agent reply to chat history
        st.session_state.chat_history.append(("agent", sms_response))
        save_chat_to_txt()  # SAVE AFTER AGENT RESPONSE

        # Show new message immediately
        st.chat_message("assistant").markdown(sms_response)

# RIGHT COLUMN - INVENTORY + TOOL CALLED
with right_col:
    st.header("🏢 Inventory Management")

    # Inventory Editable Box
    st.session_state.inventory = st.text_area(
        label="Paste available inventory here (or type 'not available')",
        value=st.session_state.inventory,
        height=300
    )

    st.header("🛠️ Tools/Agents Used")
    if st.session_state.tools_used:
        for tool_name in reversed(st.session_state.tools_used[-10:]):  # Show last 10 used
            st.success(f"✅ {tool_name}")
    else:
        st.write("No tools used yet.")
