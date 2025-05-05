import streamlit as st
from datetime import datetime
from agno_agents import QualificationTools, ToneTools, SMSFormatterTools

# Init agents
qual_tool = QualificationTools()
tone_tool = ToneTools()
sms_tool = SMSFormatterTools()

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# App UI
st.title("🏡 HomeEasy Leasing Chat")

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("What are you looking for in your next apartment?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get full user chat history only (exclude assistant)
    client_convo = "\n".join([m["content"] for m in st.session_state.messages if m["role"] == "user"])

    # STEP 1: Run Qualification Agent
    qualification_output = qual_tool.qualify_client(client_convo)

    # STEP 2: Run Tone Agent
    tone_output = tone_tool.set_tone(qualification_output)

    # STEP 3: Format for SMS Agent
    sms_input = f"""
    Based on the client's qualification and tone guidance, write a short SMS.

    Qualification Summary:
    {qualification_output}

    Tone Strategy:
    {tone_output}
    """

    sms_reply = sms_tool.format_sms(sms_input)

    with st.chat_message("assistant"):
        st.markdown(sms_reply)
    st.session_state.messages.append({"role": "assistant", "content": sms_reply})
