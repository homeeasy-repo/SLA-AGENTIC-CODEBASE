# import streamlit as st
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

# Removed get_previous_messages function - using chat_history instead

def get_client_chat_history(client_id: int) -> str:
    """Get chat history for the client from client_fub_messages table."""
    session = None
    try:
        session = get_db_connection()
        
        # Get chat history from client_fub_messages table
        query = text("""
            SELECT message
            FROM client_fub_messages 
            WHERE client_id = :client_id 
            ORDER BY created_at DESC
            LIMIT 1
        """)
        result = session.execute(query, {"client_id": client_id}).fetchone()
        
        if not result:
            print(f"No chat history found for client_id: {client_id}")
            return ""
        
        raw_message = result[0]
        
        if not raw_message:
            return ""
        
        # Return the raw message content - let the LLM parse it
        return raw_message.strip()
    
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
    """Tool function to find the single best matching inventory unit."""
    try:
        # Parse input if it's JSON
        if input_data.startswith('{'):
            data = json.loads(input_data)
            requirements = data.get('requirements', '')
            inventory = data.get('inventory', '')
            chat_history = data.get('chat_history', '')
        else:
            requirements = input_data
            inventory = ""
            chat_history = ""

        llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3  # Lower temperature for more consistent matching
        )

        matching_prompt = f"""
        You are a property matching expert. Analyze the inventory and find the SINGLE BEST UNIT that matches this client's requirements.

        CLIENT REQUIREMENTS:
        {requirements}

        CLIENT CHAT HISTORY (for context):
        {chat_history}

        AVAILABLE INVENTORY FORMAT:
        {inventory}

        INVENTORY PARSING INSTRUCTIONS:
        The inventory shows buildings with this format:
        - Building name, address, rent range, management company
        - Move-in specials (like "$500 Off first month")
        - Individual units with specific prices, bed/bath, sqft
        - Application links and details

        YOUR TASK:
        1. **Calculate Net Effective Rent**: If there are move-in specials like "$500 off first month", calculate the true monthly cost over a 12-month lease.
           Formula: (Total rent for 12 months - discount) / 12 months
           Example: $1500/month with $500 off first month = ((1500*12) - 500) / 12 = $1458.33 effective monthly

        2. **Find the SINGLE BEST UNIT** based on:
           - Budget compatibility (use net effective rent)
           - Bedroom/bathroom requirements
           - Location preferences from requirements and chat history
           - Move-in date availability
           - Special amenities mentioned in requirements

        3. **Prioritize units with move-in specials** as they offer better value

        RESPONSE FORMAT:
        Return ONLY the best matching unit with this exact structure:

        BEST MATCH FOUND: [Yes/No]
        BUILDING NAME: [Building name]
        ADDRESS: [Full address]
        UNIT DETAILS: [Bed/Bath, Sqft]
        LISTED RENT: $[original monthly rent]
        MOVE-IN SPECIAL: [special offer if any]
        NET EFFECTIVE RENT: $[calculated monthly effective rent]
        ANNUAL SAVINGS: $[total savings from specials]
        APPLICATION LINK: [direct application URL]
        GOOGLE MAPS: [maps link]
        MATCH REASONING: [Why this is the best match - 2-3 sentences]

        If no good match found, return:
        BEST MATCH FOUND: No
        REASON: [Explain why no units meet the criteria]
        """

        response = llm.invoke(matching_prompt)
        return response.content

    except Exception as e:
        return f"Error finding inventory match: {str(e)}"

def generate_client_message_tool(input_data: str) -> str:
    """Tool function to generate engaging, location-specific client message following HomeEasy guidelines."""
    try:
        # Parse input if it's JSON
        if input_data.startswith('{'):
            data = json.loads(input_data)
            property_info = data.get('property_info', '')
            client_profile = data.get('client_profile', '')
            match_found = data.get('match_found', False)
            client_name = data.get('client_name', 'there')
            chat_history = data.get('chat_history', '')
        else:
            property_info = input_data
            client_profile = ""
            match_found = True
            client_name = "there"
            chat_history = ""

        llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )

        if match_found:
            message_prompt = f"""
            You are a HomeEasy virtual sales assistant. Generate an engaging text message following these EXACT guidelines from our SLA:

            PROPERTY MATCH DETAILS:
            {property_info}

            CLIENT PROFILE & CHAT HISTORY:
            {client_profile}
            {chat_history}

            MESSAGE TONE & STYLE REQUIREMENTS:
            - Use neutral, landlord-like, non-salesy tone
            - Avoid overpromising. Never say "dream home" or "perfect match"
            - Sound human and personal, not AI-generated
            - Use Socratic framing when introducing price/availability
            - Include specific location details and property highlights
            - Create excitement with "I have a unit for you" approach

                         MESSAGE STRUCTURE:
             1. Personal greeting using client context
             2. "I have a unit for you" or similar engaging opener
             3. Specific property details: location, price, bed/bath
             4. Mention move-in special and savings if applicable
             5. Clear next step - ask if they want to see it
             6. CRITICAL: Keep under 250 characters for SMS - be very concise

            AVOID:
            - "Dream home", "perfect match", overly optimistic phrases
            - Formal language or corporate speak
            - Commission mentions
            - Availability promises without confirmation
            - Generic property descriptions

            EXAMPLE STYLE:
            "Hi [Name]! I have a unit for you - 2BR/2BA in [Location] at $[price]/mo with [special]. [Key feature]. Want to check it out? [Maps link]"

            Generate ONLY the SMS message text, nothing else.
            """
        else:
            message_prompt = f"""
            Generate a helpful text message when no perfect match was found, following HomeEasy guidelines:

            CLIENT CONTEXT:
            {client_profile}
            {chat_history}

            TONE REQUIREMENTS:
            - Neutral, professional, not salesy
            - Acknowledge limited results honestly
            - Invite feedback to refine search
            - Keep conversation open for back-and-forth
            - Under 160 characters

            STRUCTURE:
            1. Acknowledge search completion
            2. Brief explanation of limited results
            3. Ask for feedback/preferences to adjust
            4. Keep thread open for continued search

            Generate ONLY the SMS message text, nothing else.
            """

        response = llm.invoke(message_prompt)
        # Clean up the response to remove any AI-like formatting
        message = response.content.strip()
        message = re.sub(r'^["\']|["\']$', '', message)  # Remove quotes
        message = re.sub(r'\[.*?\]', '', message)  # Remove any placeholder brackets
        message = re.sub(r'Client Name', 'there', message)  # Replace placeholder names
        message = re.sub(r'\[Name\]', 'there', message)  # Replace placeholder names
        
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
            print(f"Requirements: {requirements}")
            chat_history = get_client_chat_history(client_id)
            print(f"Chat History: {chat_history}")
            inventory = get_client_markdown_content(client_id)
            print(f"Inventory: {inventory}")
            client_info = get_client_basic_info(client_id)
            print(f"Client Info: {client_info}")
            # Extract client name from client_info
            client_name = "there"  # Default
            if "Name:" in client_info:
                try:
                    name_part = client_info.split("Name: ")[1].split(",")[0].strip()
                    if name_part and name_part != "None":
                        client_name = name_part.split()[0]  # First name only
                except:
                    pass

            # Create the main prompt for the agent
            main_prompt = f"""
            You are a property matching expert. Your task is to process this client and return a final JSON response.

            CLIENT ID: {client_id}
            CLIENT NAME: {client_name}
            
            CLIENT REQUIREMENTS:
            {requirements}
            
            CLIENT CHAT HISTORY (Raw format from Follow Up Boss):
            {chat_history}
            
            CLIENT INFO:
            {client_info}
            
            AVAILABLE INVENTORY:
            {inventory}

            TOOL CALLING INSTRUCTIONS:
            1. **analyze_client_profile**: Pass JSON with requirements, chat_history, and client_info
            2. **find_inventory_match**: Pass JSON with requirements, inventory, and chat_history  
            3. **generate_client_message**: Pass JSON with property_info, client_profile, match_found, client_name, and chat_history

            CHAT HISTORY FORMAT NOTES:
            The chat history contains messages in this format:
            [timestamp] Client - Name: message text
            [timestamp] Sales Rep - Name: message text
            
            Find the LATEST CLIENT MESSAGE to understand what they need a response to.

            Your task:
            1. First analyze the client profile using the analyze_client_profile tool
            2. Then find the SINGLE BEST UNIT using the find_inventory_match tool (it will calculate net effective rent from move-in specials)
            3. Finally generate an engaging "I have a unit for you" message using the generate_client_message tool

            Based on your analysis, determine:
            - If a good property match was found (true/false)
            - Create a summary of the client
            - Generate an engaging SMS message following HomeEasy guidelines

            After using the tools, provide your final response as JSON in this exact format:
            {{
                "summary": "Brief client profile summary with key details",
                "message": "Engaging SMS message ready to send",
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