import streamlit as st
from dotenv import load_dotenv
import os
from typing import List, Dict, Any
import platform
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun, Tool
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from sqlalchemy import text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import json
import asyncio
import re

# Load environment variables
load_dotenv()

# Get environment variables
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Validate required environment variables
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Please set it in your .env file.")

# Ensure DATABASE_URL starts with postgresql://
if not DATABASE_URL.startswith('postgresql://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

def get_db_connection():
    """Create and return a database connection."""
    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        print(f"Error creating database connection: {str(e)}")
        raise

def get_client_requirements(client_id: int) -> str:
    """Fetch client requirements from the database and format them."""
    session = None
    try:
        session = get_db_connection()
        query = text("""
            SELECT 
                budget, budget_max, move_in_date, move_in_date_max,
                beds, baths, parking, pets, washer_dryer,
                neighborhood, zip, sqft, sqft_max,
                building_must_haves, unit_must_haves,
                special_needs, preference, comment,
                lease_term, section8, monthly_income,
                credit_score, amenities
            FROM requirements 
            WHERE client_id = :client_id
        """)
        result = session.execute(query, {"client_id": client_id}).fetchone()
        
        if not result:
            raise ValueError(f"No requirements found for client_id: {client_id}")
        
        # Convert result to dictionary
        req = dict(result._mapping)
        
        # Format requirements into readable text
        sections = []
        
        # Budget
        if req['budget'] or req['budget_max']:
            budget_range = f"${req['budget'] or ''} - ${req['budget_max'] or ''}"
            sections.append(f"Budget Range: {budget_range}")
        
        # Move-in Date
        if req['move_in_date']:
            date_range = f"{req['move_in_date']}"
            if req['move_in_date_max']:
                date_range += f" to {req['move_in_date_max']}"
            sections.append(f"Move-in Date: {date_range}")
        
        # Unit Details
        unit_details = []
        if req['beds']:
            unit_details.append(f"{req['beds']} Bedroom(s)")
        if req['baths']:
            unit_details.append(f"{req['baths']} Bathroom(s)")
        if unit_details:
            sections.append(f"Unit Details: {', '.join(unit_details)}")
        
        # Location
        if req['neighborhood']:
            sections.append(f"Preferred Neighborhoods: {', '.join(req['neighborhood'])}")
        if req['zip']:
            sections.append(f"Preferred ZIP Codes: {', '.join(req['zip'])}")
        
        # Additional Features
        if req['parking']:
            sections.append(f"Parking: {req['parking']}")
        if req['pets']:
            sections.append(f"Pets: {req['pets']}")
        if req['washer_dryer']:
            sections.append(f"Washer/Dryer: {req['washer_dryer']}")
        
        # Square Footage
        if req['sqft'] or req['sqft_max']:
            sqft_range = f"{req['sqft'] or ''} - {req['sqft_max'] or ''}"
            sections.append(f"Square Footage Range: {sqft_range}")
        
        # Must-Haves and Preferences
        if req['building_must_haves']:
            sections.append(f"Building Must-Haves: {req['building_must_haves']}")
        if req['unit_must_haves']:
            sections.append(f"Unit Must-Haves: {req['unit_must_haves']}")
        if req['special_needs']:
            sections.append(f"Special Needs: {req['special_needs']}")
        if req['amenities']:
            sections.append(f"Desired Amenities: {', '.join(req['amenities'])}")
        
        # Additional Requirements
        if req['lease_term']:
            sections.append(f"Lease Term: {req['lease_term']} months")
        if req['section8']:
            sections.append("Section 8: Yes")
        if req['monthly_income']:
            sections.append(f"Monthly Income: ${req['monthly_income']}")
        if req['credit_score']:
            sections.append(f"Credit Score: {req['credit_score']}")
        
        # Comments
        if req['comment']:
            sections.append(f"\nAdditional Comments:\n{req['comment']}")
        
        return "\n".join(sections)
    
    except Exception as e:
        print(f"Error fetching client requirements: {e}")
        raise
    finally:
        try:
            if session:
                session.close()
        except Exception as close_err:
            print(f"Warning: failed to close DB session: {close_err}")

def get_client_markdown_content(client_id: int) -> str:
    """Fetch markdown content from notes for the client."""
    session = None
    try:
        session = get_db_connection()
        query = text("""
            SELECT body 
            FROM note 
            WHERE client_id = :client_id 
            AND subject = 'Inventory Discovery For Client'
            ORDER BY created DESC
        """)
        results = session.execute(query, {"client_id": client_id}).fetchall()
        
        if not results:
            raise ValueError(f"No markdown content found for client_id: {client_id}")
        
        # Combine all notes, removing duplicates and empty content
        all_content = []
        seen_content = set()
        
        for result in results:
            content = result[0].strip()
            if content and content not in seen_content:
                seen_content.add(content)
                all_content.append(content)
        
        if not all_content:
            raise ValueError(f"No valid markdown content found for client_id: {client_id}")
        
        # Join all unique content with double newlines
        return "\n\n".join(all_content)
    
    except Exception as e:
        print(f"Error fetching markdown content: {e}")
        raise
    finally:
        try:
            if session:
                session.close()
        except Exception as close_err:
            print(f"Warning: failed to close DB session: {close_err}")

def get_previous_messages(client_id: int) -> list:
    """Fetch previous messages sent to the client."""
    session = None
    try:
        session = get_db_connection()
        query = text("""
            SELECT message, created
            FROM textmessage 
            WHERE client_id = :client_id 
            AND is_incoming = false
            ORDER BY created DESC
        """)
        results = session.execute(query, {"client_id": client_id}).fetchall()
        return [(row[0], row[1]) for row in results]
    except Exception as e:
        print(f"Error fetching previous messages: {e}")
        return []
    finally:
        if session:
            session.close()

def get_client_chat_history(client_id: int) -> str:
    """Get comprehensive chat history for the client."""
    session = None
    try:
        session = get_db_connection()
        
        # Get all text messages
        query = text("""
            SELECT message, created, is_incoming
            FROM textmessage 
            WHERE client_id = :client_id 
            ORDER BY created ASC
        """)
        results = session.execute(query, {"client_id": client_id}).fetchall()
        
        chat_history = []
        for row in results:
            message, created, is_incoming = row
            sender = "Client" if is_incoming else "Agent"
            chat_history.append(f"[{created}] {sender}: {message}")
        
        return "\n".join(chat_history)
    
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return ""
    finally:
        if session:
            session.close()

def get_client_basic_info(client_id: int) -> str:
    """Get client basic information."""
    session = None
    try:
        session = get_db_connection()
        query = text("""
            SELECT fullname, email, addresses, neighborhood, max_salary
            FROM public.client
            WHERE id = :client_id
        """)
        client_data = session.execute(query, {"client_id": client_id}).fetchone()
        
        if not client_data:
            return f"No client data found for ID {client_id}"
        
        return f"Name: {client_data.fullname}, Email: {client_data.email}, Address: {client_data.addresses}, Neighborhood: {client_data.neighborhood}, Max Salary: ${client_data.max_salary}"
    
    except Exception as e:
        print(f"Error fetching client info: {e}")
        return f"Error fetching client info: {str(e)}"
    finally:
        if session:
            session.close()

# Tool Functions for the Main Agent
def analyze_client_profile_tool(input_data: str) -> str:
    """Tool function to analyze client profile."""
    try:
        # Parse input if it's JSON
        if input_data.startswith('{'):
            data = json.loads(input_data)
            requirements = data.get('requirements', '')
            chat_history = data.get('chat_history', '')
            client_info = data.get('client_info', '')
        else:
            # If it's just a string, treat it as requirements
            requirements = input_data
            chat_history = ""
            client_info = ""

        llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )

        analysis_prompt = f"""
        Analyze this client's profile and provide insights about their financial status, demographics, and approval likelihood.

        CLIENT REQUIREMENTS:
        {requirements}
        
        CLIENT CHAT HISTORY:
        {chat_history}
        
        CLIENT INFO:
        {client_info}

        Provide a brief analysis focusing on:
        1. Financial stability and budget
        2. Demographics (age, family status, background)
        3. Urgency of move
        4. Likelihood of approval
        5. Communication style

        Keep response concise and factual.
        """

        response = llm.invoke(analysis_prompt)
        return response.content

    except Exception as e:
        return f"Error analyzing client profile: {str(e)}"

def find_inventory_match_tool(input_data: str) -> str:
    """Tool function to find matching inventory."""
    try:
        # Parse input if it's JSON
        if input_data.startswith('{'):
            data = json.loads(input_data)
            requirements = data.get('requirements', '')
            inventory = data.get('inventory', '')
            previous_messages = data.get('previous_messages', '')
        else:
            requirements = input_data
            inventory = ""
            previous_messages = ""

        llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )

        matching_prompt = f"""
        Find the best property match for this client from the available inventory.

        CLIENT REQUIREMENTS:
        {requirements}

        AVAILABLE INVENTORY:
        {inventory}

        PREVIOUS MESSAGES:
        {previous_messages}

        Look for:
        1. Budget compatibility
        2. Location match
        3. Property features
        4. Special offers/incentives
        5. Commission potential

        Return the best matching property with details and reasoning. If no good match, explain why.
        """

        response = llm.invoke(matching_prompt)
        return response.content

    except Exception as e:
        return f"Error finding inventory match: {str(e)}"

def generate_client_message_tool(input_data: str) -> str:
    """Tool function to generate human-like client message."""
    try:
        # Parse input if it's JSON
        if input_data.startswith('{'):
            data = json.loads(input_data)
            property_info = data.get('property_info', '')
            client_profile = data.get('client_profile', '')
            match_found = data.get('match_found', False)
        else:
            property_info = input_data
            client_profile = ""
            match_found = True

        llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )

        if match_found:
            message_prompt = f"""
            Generate a natural, human-like text message for this client about a property match.

            PROPERTY INFO:
            {property_info}

            CLIENT PROFILE:
            {client_profile}

            Requirements:
            1. Sound like a real person, not AI
            2. Keep it under 160 characters for SMS
            3. Be specific about the property
            4. Create urgency but not pushy
            5. Include call to action
            6. NO placeholder text like [insert link]
            7. NO formal language - keep it casual and friendly

            Generate ONLY the SMS message text, nothing else.
            """
        else:
            message_prompt = f"""
            Generate a natural, human-like text message for this client when no perfect match was found.

            CLIENT PROFILE:
            {client_profile}

            Requirements:
            1. Sound like a real person, not AI
            2. Keep it under 160 characters for SMS
            3. Be helpful and positive
            4. Suggest next steps
            5. NO placeholder text
            6. Keep it casual and friendly

            Generate ONLY the SMS message text, nothing else.
            """

        response = llm.invoke(message_prompt)
        # Clean up the response to remove any AI-like formatting
        message = response.content.strip()
        message = re.sub(r'^["\']|["\']$', '', message)  # Remove quotes
        message = re.sub(r'\[.*?\]', '', message)  # Remove any placeholder brackets
        
        return message

    except Exception as e:
        return f"Error generating message: {str(e)}"

class MainPropertyAgent:
    """Main agent that coordinates all property matching tasks using other agents as tools."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3  # Lower temperature for more consistent decisions
        )
        
        # Define tools for the main agent
        self.tools = [
            Tool(
                name="analyze_client_profile",
                func=analyze_client_profile_tool,
                description="Analyze client profile including financial status, demographics, and approval likelihood. Input should be client requirements, chat history, and basic info."
            ),
            Tool(
                name="find_inventory_match",
                func=find_inventory_match_tool,
                description="Find matching properties from inventory based on client requirements. Input should include requirements, inventory data, and previous messages."
            ),
            Tool(
                name="generate_client_message",
                func=generate_client_message_tool,
                description="Generate a human-like SMS message for the client. Input should include property info, client profile, and whether a match was found."
            )
        ]
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    async def process_client(self, client_id: int) -> Dict[str, Any]:
        """Main method that processes a client and returns final response."""
        
        try:
            print(f"🏠 Processing client {client_id} with Main Property Agent...")
            
            # Gather all client data once
            requirements = get_client_requirements(client_id)
            chat_history = get_client_chat_history(client_id)
            inventory = get_client_markdown_content(client_id)
            previous_messages = get_previous_messages(client_id)
            client_info = get_client_basic_info(client_id)
            
            # Format previous messages
            prev_msgs = "\n".join([f"{msg[1]}: {msg[0]}" for msg in previous_messages])
            
            # Create the main prompt for the agent
            main_prompt = f"""
            You are a property matching expert. Your task is to process this client and return a final JSON response.

            CLIENT ID: {client_id}
            
            CLIENT REQUIREMENTS:
            {requirements}
            
            CLIENT CHAT HISTORY:
            {chat_history}
            
            CLIENT INFO:
            {client_info}
            
            AVAILABLE INVENTORY:
            {inventory}
            
            PREVIOUS MESSAGES SENT:
            {prev_msgs}

            Your task:
            1. First analyze the client profile using the analyze_client_profile tool
            2. Then find matching inventory using the find_inventory_match tool  
            3. Finally generate a human-like message using the generate_client_message tool

            Based on your analysis, determine:
            - If a good property match was found (true/false)
            - Create a summary of the client
            - Generate a natural SMS message for the client

            After using the tools, provide your final response as JSON in this exact format:
            {{
                "summary": "Brief client profile summary",
                "message": "Human-like SMS message for client",
                "inventory_found": true/false
            }}
            """
            
            # Run the main agent
            response = self.agent.invoke({"input": main_prompt})
            
            # Extract JSON from response
            output = response.get("output", "")
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*"inventory_found"[\s\S]*\}', output)
            if json_match:
                try:
                    result_json = json.loads(json_match.group(0))
                    
                    # Clean up the message to ensure it's human-like
                    message = result_json.get('message', '')
                    message = re.sub(r'\[.*?\]', '', message)  # Remove brackets
                    message = re.sub(r'^["\']|["\']$', '', message)  # Remove quotes
                    message = message.strip()
                    
                    result_json['message'] = message
                    
                    return result_json
                except json.JSONDecodeError:
                    pass
            
            # Fallback response if JSON parsing fails
            inventory_found = "match" in output.lower() or "found" in output.lower()
            
            return {
                "summary": f"Client {client_id} processed - see full analysis in logs",
                "message": "Hi! Found some options for you. Let's chat about your housing needs!",
                "inventory_found": inventory_found
            }
            
        except Exception as e:
            print(f"Error processing client: {str(e)}")
            return {
                "summary": f"Error processing client {client_id}: {str(e)}",
                "message": "Hi! Let me look into some options for you and get back to you soon.",
                "inventory_found": False
            }

# Main function for external use
async def process_client(client_id: int) -> str:
    """Main entry point for processing client."""
    try:
        agent = MainPropertyAgent()
        result = await agent.process_client(client_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        error_result = {
            "summary": f"Error processing client {client_id}: {str(e)}",
            "message": "Hi! Let me look into some options for you and get back to you soon.",
            "inventory_found": False
        }
        return json.dumps(error_result, indent=2)

if __name__ == "__main__":
    async def main():
        """Test the main agent."""
        try:
            client_id = 691481
            result = await process_client(client_id)
            print("Final Result:")
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}")

    asyncio.run(main())