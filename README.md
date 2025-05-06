## To Run the google-adk Agent 

# venv is being used here Python Version : `Python 3.11.9`

# First Install all requirements by running command:
    `pip install -r requirements.txt`

1. Check the multi_tool_agent folder
    Navigate to the folder 
        `multi_tool_agent`
    Command for terminal Run:
        `adk run .`

2. Run on WEB UI
    Command: 
        `adk web`

# Agno Agnet

There are two files `agno_agents.py` in which we have all the agents defined & `agno_test.py` which works with streamlit and is being used to interact with the agent.

Command to Run:
    `streamlit run agno_test.py`

## Production Unstable ##

Here is repo contain inventory agent code which is deployed on streamlit.

Go to streamlit --> Find the organization --> open the inventory agent app.