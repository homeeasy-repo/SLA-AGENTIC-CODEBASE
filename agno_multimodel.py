import asyncio
from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.team.team import Team
from dotenv import load_dotenv
import os
from sqlalchemy import text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import json

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

client_profiler = Agent(
    name="Client Profiler",
    role="Analyze client profile and requirements",
    model=Gemini(id=GENAI_MODEL, api_key=GOOGLE_API_KEY),
    add_name_to_instructions=True,
    instructions=dedent("""
    You are a client profiling expert.
    Your task is to analyze client data including:
    1. Financial status (budget, income, credit score)
    2. Demographics (location, family status)
    3. Employment status
    4. Move-in timeline
    5. Criminal history (if available)
    6. Special requirements or preferences
    
    Provide a comprehensive profile that includes:
    - Financial stability assessment
    - Likelihood of approval
    - Urgency of move
    - Special considerations
    - Risk factors
    
    Use the client requirements and chat history to make informed judgments.
    Also analyze the available inventory to understand what properties are being considered.
    """),
)

inventory_matcher = Agent(
    name="Inventory Matcher",
    role="Match client requirements with available inventory",
    model=Gemini(id=GENAI_MODEL, api_key=GOOGLE_API_KEY),
    add_name_to_instructions=True,
    instructions=dedent("""
    You are an inventory matching expert.
    Your task is to:
    1. Analyze client requirements (budget, location, move-in date)
    2. Review available inventory from the notes
    3. Check previous messages to avoid suggesting the same property
    4. Identify properties that match client needs and haven't been suggested before
    5. Consider special offers and commission opportunities
    
    For each matching property, provide:
    - Property details
    - Location analysis
    - Special offers
    - Commission potential
    - Confirmation that this property hasn't been suggested before
    
    Select at least one property that best matches the client's needs.
    Focus on properties mentioned in the inventory notes.
    If a property was previously suggested, find a different one.
    """),
)

team_coordinator = Agent(
    name="Team Coordinator",
    role="Coordinate between client profile and inventory matching",
    model=Gemini(id=GENAI_MODEL, api_key=GOOGLE_API_KEY),
    add_name_to_instructions=True,
    instructions=dedent("""
    You are a team coordination expert.
    Your task is to:
    1. Review client profile analysis
    2. Evaluate inventory matches
    3. Make final property recommendations
    4. Generate a concise SMS message for the client
    
    The SMS message should include ONLY:
    - Property type and location
    - Price
    - Move-in date
    - Any special offers or incentives
    
    Keep the message under 160 characters and use SMS-friendly language.
    No formal greetings, signatures, or phone numbers needed.
    DO NOT ask for additional information.
    DO NOT mention income, credit score, or criminal history.
    Focus only on the property details.
    
    Example format:
    "Perfect 1BR in Pilsen, $1300/mo. Move-in May 21."
    
    DO NOT include any text before or after the message.
    DO NOT include any explanations or introductions.
    DO NOT include phrases like "Based on the analysis" or "I recommend".
    ONLY output the message.
    """),
)

json_formatter = Agent(
    name="JSON Formatter",
    role="Format the agent team's response into JSON",
    model=Gemini(id=GENAI_MODEL, api_key=GOOGLE_API_KEY),
    add_name_to_instructions=True,
    instructions=dedent("""
    You are a JSON formatting expert.
    Your task is to:
    1. Take the agent team's message
    2. Format it into a JSON object with a message field
    
    Your response MUST be in this exact JSON format:
    {
        "message": "[agent team's message]"
    }
    
    Example input: "Hi Colin Grasley, we found a great 1-bedroom apartment in University of Illinois Chicago area for $1500. Move-in date is 2025-06-25."
    Example output: {
        "message": "Hi Colin Grasley, we found a great 1-bedroom apartment in University of Illinois Chicago area for $1500. Move-in date is 2025-06-25."
    }
    
    DO NOT include any text before or after the JSON object.
    ONLY output the JSON object.
    """),
)

agent_team = Team(
    name="Property Matching Team",
    mode="collaborate",
    model=Gemini(id=GENAI_MODEL, api_key=GOOGLE_API_KEY),
    members=[
        client_profiler,
        inventory_matcher,
        team_coordinator,
    ],
    instructions=[
        "You are a property matching team.",
        "Work together to find the best property match for the client.",
        "Provide a single, concise SMS message with the best property match.",
        "No detailed analysis or explanations needed in the final message.",
        "the message must have hi {client_name} in it"
    ],
    success_criteria="The team has found the best property match and generated a concise SMS message in json.",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    show_members_responses=True,
)

def get_db_connection():
    """Create and return a database connection."""
    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        return Session()
    except Exception as e:
        print(f"Error creating database connection: {str(e)}")
        raise

def format_client_requirements(requirements_str: str) -> str:
    """Format client requirements into a readable string."""
    try:
        # If requirements is already a string, return it
        if isinstance(requirements_str, str):
            return requirements_str
            
        # If requirements is a dictionary, format it
        if isinstance(requirements_str, dict):
            req_list = []
            
            # Basic requirements
            if requirements_str.get('budget') or requirements_str.get('budget_max'):
                budget_range = f"${requirements_str.get('budget', '')}-${requirements_str.get('budget_max', '')}"
                req_list.append(f"- Budget: {budget_range} per month")
            
            if requirements_str.get('move_in_date'):
                req_list.append(f"- Move-in Date: {requirements_str['move_in_date']}")
            
            if requirements_str.get('beds') or requirements_str.get('baths'):
                unit_details = []
                if requirements_str.get('beds'):
                    unit_details.append(f"{requirements_str['beds']} bedroom")
                if requirements_str.get('baths'):
                    unit_details.append(f"{requirements_str['baths']} bathroom")
                req_list.append(f"- Unit Details: {' '.join(unit_details)}")
            
            # Location preferences
            if requirements_str.get('neighborhood'):
                neighborhoods = ', '.join(requirements_str['neighborhood'])
                req_list.append(f"- Location: {neighborhoods}")
            
            if requirements_str.get('zip'):
                zip_codes = ', '.join(requirements_str['zip'])
                req_list.append(f"- Zip Codes: {zip_codes}")
            
            # Additional preferences
            if requirements_str.get('parking'):
                req_list.append(f"- Parking: {requirements_str['parking']}")
            
            if requirements_str.get('pets'):
                req_list.append(f"- Pets: {requirements_str['pets']}")
            
            if requirements_str.get('washer_dryer'):
                req_list.append(f"- Washer/Dryer: {requirements_str['washer_dryer']}")
            
            if requirements_str.get('sqft') or requirements_str.get('sqft_max'):
                sqft_range = f"{requirements_str.get('sqft', '')}-{requirements_str.get('sqft_max', '')}"
                req_list.append(f"- Square Footage: {sqft_range}")
            
            if requirements_str.get('amenities'):
                amenities = ', '.join(requirements_str['amenities'])
                req_list.append(f"- Amenities: {amenities}")
            
            if requirements_str.get('building_must_haves'):
                req_list.append(f"- Building Must Haves: {requirements_str['building_must_haves']}")
            
            if requirements_str.get('unit_must_haves'):
                req_list.append(f"- Unit Must Haves: {requirements_str['unit_must_haves']}")
            
            if requirements_str.get('special_needs'):
                req_list.append(f"- Special Needs: {requirements_str['special_needs']}")
            
            if requirements_str.get('preference'):
                req_list.append(f"- Preferences: {requirements_str['preference']}")
            
            if requirements_str.get('comment'):
                req_list.append(f"- Additional Comments: {requirements_str['comment']}")
            
            return "\n".join(req_list)
            
        return str(requirements_str)
    except Exception as e:
        print(f"Error formatting requirements: {str(e)}")
        return str(requirements_str)

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

async def format_to_json(message: str) -> str:
    """Format the agent team's response into JSON."""
    if not message:
        return json.dumps({"message": ""})
    return json.dumps({"message": message})

async def process_client(client_id: int):
    """Process client data and run the agent team analysis."""
    session = None
    try:
        # Get database connection
        session = get_db_connection()
        
        # Get all client data
        client_requirements = get_client_requirements(client_id)
        client_chat_history = get_client_markdown_content(client_id)
        previous_messages = get_previous_messages(client_id)
        
        # Get client data using the session
        query = text("""
            SELECT 
                neighborhood,
                addresses,
                email,
                max_salary,
                fullname
            FROM public.client
            WHERE id = :client_id
        """)
        client_data = session.execute(query, {"client_id": client_id}).fetchone()
        
        if not client_data:
            raise ValueError(f"No client data found for client_id: {client_id}")
        
        # Format client data
        client_info = f"""
        CLIENT PROFILE:
        --------------
        Name: {client_data.fullname}
        Email: {client_data.email}
        Address: {client_data.addresses}
        Neighborhood: {client_data.neighborhood}
        Max Salary: ${client_data.max_salary}
        """
        
        # Format previous messages
        previous_messages_text = "\n".join([
            f"Message sent on {msg[1]}: {msg[0]}"
            for msg in previous_messages
        ])
        
        # Format the requirements for better readability
        formatted_requirements = format_client_requirements(client_requirements)
        
        # Prepare the comprehensive message for the agents
        message = f"""
        CLIENT PROFILE ANALYSIS REQUEST
        ==============================
        
        CLIENT INFORMATION:
        {client_info}
        
        CLIENT REQUIREMENTS:
        {formatted_requirements}
        
        CLIENT CHAT HISTORY AND INVENTORY:
        {client_chat_history}
        
        PREVIOUS MESSAGES:
        {previous_messages_text}
        
        TASK INSTRUCTIONS:
        1. Client Profiler: Analyze the client's profile, requirements, and chat history to create a detailed client profile.
        2. Inventory Matcher: Review available properties and match them with client requirements, ensuring no duplicate suggestions.
        3. Team Coordinator: Evaluate matches and select the best property for the client.
        
        FOCUS AREAS:
        - Financial stability and approval likelihood
        - Urgency of move
        - Location preferences
        - Special requirements
        - Property matching criteria
        - Commission potential
        - Special offers and incentives
        - Avoid suggesting previously sent properties
        
        Please provide:
        1. Detailed client profile analysis
        2. Property matches with reasoning
        3. Best property recommendation
        4. Draft client communication
        """
        
        # Store the original print function
        original_print = print
        team_responses = []

        # Custom print function to capture responses
        def custom_print(*args, **kwargs):
            response = " ".join(str(arg) for arg in args)
            team_responses.append(response)
            original_print(*args, **kwargs)

        # Replace print with custom function
        import builtins
        builtins.print = custom_print

        try:
            # Run the agent team analysis
            await agent_team.aprint_response(
                message=message,
                stream=True,
                stream_intermediate_steps=True,
            )
            
            # Find the Team Coordinator's response
            team_message = None
            for response in reversed(team_responses):
                if "Response" in response and "Hi" in response:
                    # Extract the message between the response markers
                    start_idx = response.find("Hi")
                    if start_idx >= 0:
                        team_message = response[start_idx:].strip()
                        # Remove any trailing markers or extra text
                        end_idx = team_message.find("┗")
                        if end_idx >= 0:
                            team_message = team_message[:end_idx].strip()
                        break
            
            # Format the response into JSON
            json_response = await format_to_json(team_message)
            print(f"Final Message: {json_response}")
            
        finally:
            # Restore original print function
            builtins.print = original_print
        
    except Exception as e:
        print(f"Error processing client {client_id}: {str(e)}")
        raise
    finally:
        if session:
            session.close()

if __name__ == "__main__":
    async def main():
        """Main function to run the client processing."""
        try:
            # Example usage with client_id
            client_id = 689283  # Replace with actual client ID
            
            # Process the client
            await process_client(client_id)
            
        except Exception as e:
            print(f"Error in main function: {str(e)}")
            raise

    # Run the async main function
    asyncio.run(main())









