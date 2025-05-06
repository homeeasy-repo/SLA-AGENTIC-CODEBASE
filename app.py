# import streamlit as st
# import logging
# from agents import main_agent  # Import MainAgent from agent.py

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Streamlit page configuration
# st.set_page_config(page_title="HomeEasy Sales Agent", page_icon="🏡", layout="wide")

# # Initialize session state variables
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "inventory" not in st.session_state:
#     st.session_state.inventory = "not available"
# if "notes" not in st.session_state:
#     st.session_state.notes = ""
# if "tools_used" not in st.session_state:
#     st.session_state.tools_used = []

# # Function to save chat history to a TXT file
# def save_chat_to_txt():
#     """Save the chat history to a TXT file."""
#     try:
#         with open("chat_history.txt", "w", encoding="utf-8") as f:
#             for sender, message in st.session_state.chat_history:
#                 if sender == "client":
#                     f.write(f"Client: {message}\n")
#                 else:
#                     f.write(f"Agent: {message}\n")
#         logger.info("Chat history saved to chat_history.txt")
#     except Exception as e:
#         logger.error("Error saving chat history: %s", str(e))
#         st.error(f"Failed to save chat history: {str(e)}")

# # Initialize Main Agent
# @st.cache_resource
# def load_agent():
#     logger.info("Loading MainAgent")
#     return main_agent

# main_agent = load_agent()

# # Layout Columns
# left_col, center_col, right_col = st.columns([1, 2, 1])

# # LEFT COLUMN - NOTES SECTION
# with left_col:
#     st.header("📝 Notes")
#     st.session_state.notes = st.text_area(
#         label="Write your observations or mistakes here",
#         value=st.session_state.notes,
#         height=400,
#         placeholder="Enter notes about the conversation or any issues observed..."
#     )

# # CENTER COLUMN - CHAT INTERFACE
# with center_col:
#     st.title("🏡 HomeEasy Sales Bot")
#     st.caption("Talk naturally like a client. Agent will respond like a real leasing consultant!")

#     # Show Chat History
#     for sender, message in st.session_state.chat_history:
#         if sender == "client":
#             st.chat_message("user").markdown(message)
#         else:
#             st.chat_message("assistant").markdown(message)

#     # Input Box for Client Message
#     client_input = st.chat_input("Type your message as the client...")

#     if client_input:
#         # Validate client input
#         if not client_input.strip():
#             st.warning("Please enter a valid message.")
#             logger.warning("Empty client input received")
#         else:
#             # Add client message to history
#             st.session_state.chat_history.append(("client", client_input))
#             save_chat_to_txt()  # Save after client message

#             # Prepare context for MainAgent
#             full_chat_history = "\n".join(
#                 [f"{'Client' if sender == 'client' else 'Agent'}: {message}" 
#                  for sender, message in st.session_state.chat_history]
#             )

#             # Validate inventory
#             inventory = st.session_state.inventory.strip() if st.session_state.inventory else "not available"
#             if not inventory:
#                 inventory = "not available"
#                 st.session_state.inventory = inventory
#                 logger.warning("Inventory was empty; set to 'not available'")

#             # Format the input for MainAgent
#             full_context = {
#                 "chat_history": full_chat_history,
#                 "inventory_list": inventory
#             }

#             try:
#                 logger.info("Sending context to MainAgent: %s", full_context)
#                 # Call MainAgent with context
#                 sms_response = main_agent.process_query(full_context)

#                 # Log the tools used
#                 st.session_state.tools_used.append("MainAgent + SMSFormatterAgent")

#                 # Add agent reply to chat history
#                 st.session_state.chat_history.append(("agent", sms_response))
#                 save_chat_to_txt()  # Save after agent response

#                 # Show new message immediately
#                 st.chat_message("assistant").markdown(sms_response)
#                 logger.info("Agent response: %s", sms_response)
#             except Exception as e:
#                 error_message = f"Error processing message: {str(e)}"
#                 st.error(error_message)
#                 st.session_state.chat_history.append(("agent", "Sorry, I'm having trouble. Could you clarify what you're looking for?"))
#                 save_chat_to_txt()
#                 logger.error("Error in MainAgent: %s", str(e))

# # RIGHT COLUMN - INVENTORY + TOOLS USED
# with right_col:
#     st.header("🏢 Inventory Management")
#     # Inventory Editable Box
#     st.session_state.inventory = st.text_area(
#         label="Paste available inventory here (or type 'not available')",
#         value=st.session_state.inventory,
#         height=300,
#         placeholder="e.g., Property A: 2-bed, $2000/month; Property B: 1-bed, $1500/month"
#     )



#     st.header("🛠️ Tools/Agents Used")
#     if st.session_state.tools_used:
#         for tool_name in reversed(st.session_state.tools_used[-10:]):  # Show last 10 used
#             st.success(f"{tool_name}")
#     else:
#         st.write("No tools used yet.")


#### Inventory Page #####
from agents import main_agent 

def main():
    print("🏡 Welcome to HomeEasy Sales Bot!")
    print("Type 'exit' anytime to quit.\n")

    while True:
        print("\n📩 Enter the conversation/chat history (client messages):")
        chat_history = input("> ")

        if chat_history.lower() in ["exit", "quit"]:
            break

        print("\n🏢 Enter the available inventory (property listings):")
        inventory_list = input("> ")

        if inventory_list.lower() in ["exit", "quit"]:
            break

        # Pack the input into a dict
        full_context = {
            "chat_history": chat_history,
            "inventory_list": inventory_list
        }

        # Call MainAgent
        sms_response = main_agent.process_query(full_context)

        print("\n📨 Final SMS to Client:")
        print(sms_response)
        print("\n" + "-"*70)

if __name__ == "__main__":
    main()
