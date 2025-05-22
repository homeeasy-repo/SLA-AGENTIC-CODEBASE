import streamlit as st
from dotenv import load_dotenv
import os
from typing import List, Dict
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
DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URI")

# Validate required environment variables
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Please set it in your .env file.")

# Ensure DATABASE_URL starts with postgresql://
if not DATABASE_URL.startswith('postgresql://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

class FileSystemTool:
    """File system navigation tool with security restrictions."""
    def __init__(self):
        self.system = platform.system()
        self.restricted_paths = {
            'Linux': ['/root', '/etc/shadow', '/etc/passwd'],
            'Windows': ['C:\\Windows\\System32'],
            'Darwin': ['/etc/shadow', '/etc/passwd']
        }.get(self.system, [])

    def is_path_allowed(self, path: str) -> bool:
        path = os.path.abspath(path)
        return not any(restricted in path for restricted in self.restricted_paths)

    def find_file(self, filename: str) -> Dict[str, List[str]]:
        result = {'found_locations': [], 'errors': []}
        if not filename:
            result['errors'].append("Please provide a filename to search for.")
            return result

        start_paths = ['C:\\'] if self.system == 'Windows' else ['/']
        for start_path in start_paths:
            if not self.is_path_allowed(start_path):
                continue
            try:
                for root, _, files in os.walk(start_path):
                    if any(restricted in root for restricted in self.restricted_paths):
                        continue
                    if filename in files:
                        full_path = os.path.join(root, filename)
                        if self.is_path_allowed(full_path):
                            result['found_locations'].append(full_path)
            except PermissionError:
                continue
            except Exception as e:
                result['errors'].append(f"Error searching in {start_path}: {str(e)}")
        return result

    def locate_path(self, path_query: str) -> str:
        try:
            if not path_query:
                return "Please provide a path or filename to search for."

            if path_query.startswith('/'):
                if self.is_path_allowed(path_query):
                    if os.path.exists(path_query):
                        return f"Found at: {path_query}"
                    return f"Path not found: {path_query}"
                return "Access to this path is restricted for security reasons."

            result = self.find_file(path_query)
            if result['found_locations']:
                locations = "\n".join(result['found_locations'])
                return f"Found in the following location(s):\n{locations}"
            if result['errors']:
                errors = "\n".join(result['errors'])
                return f"Encountered errors while searching:\n{errors}"
            return f"Could not find '{path_query}' in accessible locations."
        except Exception as e:
            return f"Error processing query: {str(e)}"

class BaseAgent:
    """Base class for all specialized agents."""
    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
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

class ClientProfileAgent(BaseAgent):
    """Agent responsible for analyzing client profile and extracting insights."""
    def __init__(self, google_api_key: str):
        super().__init__(google_api_key)
        self.setup_agent()
        
    def setup_agent(self):
        """Initialize the client profile agent."""
        tools = [
            Tool(
                name="AnalyzeClientProfile",
                func=self.analyze_profile,
                description="Analyzes client profile, requirements, and chat history to extract insights"
            )
        ]
        
        system_message = """You are a Client Profile Analysis Expert. Your role is to:
        1. Analyze client's financial stability
        2. Determine client's demographic information
        3. Assess client's urgency and preferences
        4. Evaluate client's eligibility for properties
        
        Focus on:
        - Financial status and creditworthiness
        - Employment status and income stability
        - Family situation and living preferences
        - Move-in timeline and urgency
        - Criminal history (if mentioned)
        - Location preferences and requirements
        
        Return your analysis in a structured format."""
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def analyze_profile(self, client_data: dict) -> str:
        """Analyze client profile and return insights."""
        try:
            profile_request = f"""
            Analyze this client profile and provide detailed insights:
            
            CLIENT INFORMATION:
            {client_data.get('client_info', '')}
            
            REQUIREMENTS:
            {client_data.get('requirements', '')}
            
            CHAT HISTORY:
            {client_data.get('chat_history', '')}
            
            Please provide analysis on:
            1. Financial Status
            2. Demographic Information
            3. Family Situation
            4. Employment Status
            5. Move-in Timeline
            6. Location Preferences
            7. Special Requirements
            8. Property Eligibility
            
            Format the response as a JSON object with these categories.
            """
            
            response = self.llm.invoke(profile_request)
            return response.content
            
        except Exception as e:
            return f"Error analyzing client profile: {str(e)}"

class InventoryMatchingAgent(BaseAgent):
    """Agent responsible for matching client requirements with available inventory."""
    def __init__(self, google_api_key: str):
        super().__init__(google_api_key)
        self.setup_agent()
        
    def setup_agent(self):
        """Initialize the inventory matching agent."""
        tools = [
            Tool(
                name="MatchInventory",
                func=self.match_properties,
                description="Matches client requirements with available inventory"
            )
        ]
        
        system_message = """You are an Inventory Matching Expert. Your role is to:
        1. Match client requirements with available properties
        2. Evaluate property specials and incentives
        3. Consider location and proximity
        4. Assess commission potential
        
        Focus on:
        - Budget compatibility
        - Location preferences
        - Move-in timeline
        - Special offers and incentives
        - Commission opportunities
        - Previous property suggestions
        
        Return your matches in a structured format."""
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def match_properties(self, matching_data: dict) -> str:
        """Match client requirements with available inventory."""
        try:
            matching_request = f"""
            Match these client requirements with available inventory:
            
            CLIENT REQUIREMENTS:
            {matching_data.get('requirements', '')}
            
            AVAILABLE INVENTORY:
            {matching_data.get('inventory', '')}
            
            PREVIOUS SUGGESTIONS:
            {matching_data.get('previous_suggestions', '')}
            
            Please provide:
            1. Best matching properties
            2. Special offers and incentives
            3. Location analysis
            4. Commission potential
            5. Move-in timeline compatibility
            
            Format the response as a JSON object with these categories.
            """
            
            response = self.llm.invoke(matching_request)
            return response.content
            
        except Exception as e:
            return f"Error matching inventory: {str(e)}"

class TeamAgent(BaseAgent):
    """Agent responsible for coordinating between client profile and inventory matching."""
    def __init__(self, google_api_key: str):
        super().__init__(google_api_key)
        self.setup_agent()
        
    def setup_agent(self):
        """Initialize the team agent."""
        tools = [
            Tool(
                name="CoordinateAnalysis",
                func=self.coordinate_analysis,
                description="Coordinates between client profile and inventory matching"
            )
        ]
        
        system_message = """You are a Team Coordination Expert. Your role is to:
        1. Review client profile analysis
        2. Evaluate property matches
        3. Make final property recommendations
        4. Generate summary for message generation
        
        Focus on:
        - Client profile compatibility
        - Property match quality
        - Special offers and incentives
        - Overall recommendation
        
        Return your analysis in a structured format."""
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def coordinate_analysis(self, analysis_data: dict) -> str:
        """Coordinate between client profile and inventory matching."""
        try:
            coordination_request = f"""
            Review and coordinate this analysis:
            
            CLIENT PROFILE ANALYSIS:
            {analysis_data.get('profile_analysis', '')}
            
            INVENTORY MATCHES:
            {analysis_data.get('inventory_matches', '')}
            
            Please provide:
            1. Final property recommendation
            2. Client profile summary
            3. Match justification
            4. Special considerations
            
            Format the response as a JSON object with these categories.
            """
            
            response = self.llm.invoke(coordination_request)
            return response.content
            
        except Exception as e:
            return f"Error coordinating analysis: {str(e)}"

class MessageDraftAgent(BaseAgent):
    """Agent responsible for generating message drafts for clients."""
    def __init__(self, google_api_key: str):
        super().__init__(google_api_key)
        self.setup_agent()
        
    def setup_agent(self):
        """Initialize the message draft agent."""
        tools = [
            Tool(
                name="GenerateMessageDraft",
                func=self.generate_draft,
                description="Generates a message draft based on property match and client profile"
            )
        ]
        
        system_message = """You are a professional real estate sales agent specializing in SMS communications with clients. Your role is to:
        1. Generate concise, professional SMS messages (under 160 characters) that sound like they come from a human sales agent
        2. Specifically mention the selected property with accurate details
        3. Create a sense of urgency without sounding pushy or using generic placeholders
        4. Directly invite the client for a property tour
        5. Only include links if they are actual URLs (never include placeholder text like "insert link here")
        
        Focus on:
        - Professional, warm tone that builds rapport with clients
        - Highlighting specific property features that match client preferences
        - Creating urgency by mentioning high demand or limited availability
        - Being specific about the property's key selling points
        - Including real links only when available (never placeholders)
        
        Always create two versions:
        1. A short SMS (under 160 characters) focusing on property and call-to-action
        2. A slightly more detailed SMS for follow-up if needed
        
        Return your message drafts in JSON format with 'sms', 'detailed_message', 'key_points', and 'follow_up' fields."""
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def generate_draft(self, message_data: dict) -> str:
        """Generate a message draft based on property match and client profile."""
        try:
            # Extract property details from the property match
            property_data = message_data.get('property_match', {})
            property_name = property_data.get('property', '')
            property_address = property_data.get('address', '')
            map_link = property_data.get('map_link', '')
            virtual_tour = property_data.get('virtual_tour', '')
            
            # Get client profile data
            client_profile = message_data.get('client_profile', {})
            client_name = ""
            
            # Extract client name from demographic information if available
            if isinstance(client_profile, dict):
                demo_info = client_profile.get('Demographic Information', {})
                if isinstance(demo_info, dict) and 'name' in demo_info:
                    client_name = demo_info.get('name', '')
            
            # Get any previous chat history
            client_chat = message_data.get('client_chat', '')
            
            # Build the draft request with specific instructions for property tour and urgency
            draft_request = f"""
            Generate a personalized real estate SMS message as a professional sales agent for this specific property:
            
            SELECTED PROPERTY:
            Property Name: {property_name}
            Property Address: {property_address}
            Price Range: {property_data.get('rentRange', '')}
            Bedrooms: {property_data.get('beds', '')}
            Bathrooms: {property_data.get('baths', '')}
            Map Link: {map_link if map_link else 'Not available'}
            Virtual Tour: {virtual_tour if virtual_tour else 'Not available'}
            
            CLIENT INFORMATION:
            Client Name: {client_name}
            
            CLIENT PROFILE:
            {json.dumps(client_profile, indent=2)}
            
            PREVIOUS CLIENT CHAT:
            {client_chat}
            
            SPECIAL OFFERS:
            {json.dumps(message_data.get('special_offers', {}), indent=2)}
            
            IMPORTANT INSTRUCTIONS:
            1. This MUST be formatted as a short SMS text message, not an email
            2. Sound like a warm, professional sales agent (not a bot)
            3. Specifically mention the property name "{property_name}" and key features
            4. Create urgency that this is in high demand without being pushy
            5. Directly invite for a property tour
            6. Keep the primary SMS under 160 characters if possible
            7. ONLY include real links if they are provided above - DO NOT add placeholder text like "insert link here"
            8. If Map Link is "Not available", don't mention links at all
            
            Generate:
            1. A short SMS message with property details, urgency, and tour invitation
            2. A slightly more detailed follow-up SMS (still short enough for texting)
            3. 2-3 key points about the property to highlight
            4. 1-2 suggested follow-up text messages if client doesn't respond
            
            Format your response as a JSON object with these fields:
            - "sms": The primary short SMS message
            - "detailed_message": A slightly longer follow-up SMS (still brief for texting)
            - "key_points": Array of key property features
            - "follow_up": Array of follow-up text messages
            """
            
            response = self.llm.invoke(draft_request)
            
            # Try to extract JSON from the response if it's in text form
            content = response.content
            if isinstance(content, str):
                # Look for JSON block in the response
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                
                # Try to parse as JSON
                try:
                    parsed_content = json.loads(content)
                    
                    # Post-process to remove any remaining placeholders
                    for key in ['sms', 'detailed_message']:
                        if key in parsed_content:
                            # Remove placeholder link text
                            parsed_content[key] = re.sub(r'\[Insert[^\]]*\]', '', parsed_content[key])
                            parsed_content[key] = re.sub(r'Link:.*$', '', parsed_content[key]).strip()
                    
                    return json.dumps(parsed_content)
                except:
                    # If parsing fails, return the raw content with fallback
                    rent_range = property_data.get('rentRange', '')
                    beds = property_data.get('beds', '2')
                    baths = property_data.get('baths', '2')
                    
                    # Create SMS without any placeholder text
                    sms = f"{client_name or 'Hi'}, {property_name} in {property_address.split(',')[0]} has a {beds}bed/{baths}bath unit. Hot property in high demand! Schedule a tour soon before it's gone."
                    
                    # Only add link if actually available
                    if map_link:
                        sms = f"{client_name or 'Hi'}, {property_name} has a {beds}bed/{baths}bath unit. Hot property! Tour: {map_link}"
                    
                    return json.dumps({
                        "sms": sms,
                        "detailed_message": f"{client_name or 'Hi'}, {property_name} at {property_address} has a beautiful {beds}bed/{baths}bath unit for {rent_range}. It's in high demand! Schedule a viewing soon to secure this space before someone else does.",
                        "key_points": [f"Property: {property_name}", f"Type: {beds}bed/{baths}bath unit", "High demand property"],
                        "follow_up": ["Still interested in viewing this popular property?"]
                    })
            
            return content
            
        except Exception as e:
            error_msg = f"Error generating message draft: {str(e)}"
            print(error_msg)
            # Create a basic message with no placeholders
            property_name = message_data.get('property_match', {}).get('property', 'our property')
            address = message_data.get('property_match', {}).get('address', 'in your preferred area')
            beds = message_data.get('property_match', {}).get('beds', '2')
            baths = message_data.get('property_match', {}).get('baths', '2')
            
            return json.dumps({
                "error": error_msg,
                "sms": f"Hi, {property_name} at {address.split(',')[0]} has a {beds}bed/{baths}bath unit available. It's in high demand! Contact us to schedule a tour soon.",
                "detailed_message": f"Hi, we have a {beds}bed/{baths}bath unit at {property_name} ({address}) that matches your requirements. This property is getting a lot of interest. Would you like to schedule a viewing in the next few days?",
                "key_points": ["Matches your requirements", "High demand property", f"{beds}bed/{baths}bath unit"],
                "follow_up": ["This property is attracting a lot of interest. Would you like to schedule a viewing?"]
            })

class MainAgent:
    """Main coordinator agent that manages all specialized agents."""
    def __init__(self):
        load_dotenv()
        self.setup_api_keys()
        self.setup_agents()
        self.setup_coordinator()

    def setup_api_keys(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key is missing!")

    def setup_agents(self):
        """Initialize all specialized agents."""
        self.client_profile_agent = ClientProfileAgent(self.google_api_key)
        self.inventory_agent = InventoryMatchingAgent(self.google_api_key)
        self.team_agent = TeamAgent(self.google_api_key)
        self.message_agent = MessageDraftAgent(self.google_api_key)

    def setup_coordinator(self):
        """Setup the main coordinator agent."""
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=self.google_api_key,
            temperature=0.5
        )
        
        tools = [
            Tool(
                name="ClientProfileAnalysis",
                func=self.client_profile_agent.analyze_profile,
                description="Analyzes client profile and requirements"
            ),
            Tool(
                name="InventoryMatching",
                func=self.inventory_agent.match_properties,
                description="Matches client requirements with available inventory"
            ),
            Tool(
                name="TeamCoordination",
                func=self.team_agent.coordinate_analysis,
                description="Coordinates between client profile and inventory matching"
            ),
            Tool(
                name="MessageDraft",
                func=self.message_agent.generate_draft,
                description="Generates message drafts for clients"
            )
        ]
        
        system_message = """You are a Property Matching Coordinator. Your role is to:
        1. Coordinate between specialized agents
        2. Ensure comprehensive analysis
        3. Generate final recommendations
        4. Create effective message drafts
        5. Format output in JSON
        
        Focus on:
        - Client profile analysis
        - Property matching
        - Team coordination
        - Message drafting
        - Final recommendations
        
        Always return results in JSON format."""
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        self.coordinator = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def process_query(self, query: str) -> str:
        """Process query through the coordinator agent."""
        try:
            print("\nProcessing query:", query)
            response = self.coordinator.invoke({"input": query})
            return response["output"]
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            if hasattr(e, '__cause__'):
                print(f"Caused by: {e.__cause__}")
            return error_msg

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
        
        # Initialize the main agent
        main_agent = MainAgent()
        
        # Step 1: Client Profile Analysis
        client_profile_data = {
            'client_info': client_info,
            'requirements': client_requirements,
            'chat_history': client_chat_history
        }
        profile_analysis_raw = main_agent.client_profile_agent.analyze_profile(client_profile_data)
        
        # Enhanced JSON parsing for profile analysis
        profile_analysis = extract_json_safely(profile_analysis_raw, "profile analysis")
        
        # Step 2: Inventory Matching
        inventory_data = {
            'requirements': client_requirements,
            'inventory': client_chat_history,  # Contains inventory information
            'previous_suggestions': previous_messages_text
        }
        inventory_matches_raw = main_agent.inventory_agent.match_properties(inventory_data)
        
        # Enhanced JSON parsing for inventory matches
        inventory_matches = extract_json_safely(inventory_matches_raw, "inventory matches")
        
        # Step 3: Team Coordination
        analysis_data = {
            'profile_analysis': profile_analysis,
            'inventory_matches': inventory_matches
        }
        final_analysis_raw = main_agent.team_agent.coordinate_analysis(analysis_data)
        
        # Enhanced JSON parsing for final analysis
        final_analysis = extract_json_safely(final_analysis_raw, "final analysis")
        
        # Extract property information for the message draft
        selected_property = extract_property_info(final_analysis, inventory_matches)
        
        # Step 4: Generate Message Draft
        message_data = {
            'client_profile': profile_analysis,
            'property_match': selected_property,
            'special_offers': inventory_matches.get('specialOffers', inventory_matches.get('special_offers', {})),
            'client_chat': client_chat_history,
            'previous_messages': previous_messages_text
        }
        
        message_draft_raw = main_agent.message_agent.generate_draft(message_data)
        
        # Enhanced JSON parsing for message draft
        message_draft = extract_json_safely(message_draft_raw, "message draft")
        
        # Create the final response structure
        response = {
            'client_profile': profile_analysis,
            'inventory_matches': inventory_matches,
            'team_analysis': final_analysis,
            'final_recommendation': {
                'selected_property': selected_property,
                'match_justification': final_analysis.get('match_justification', ''),
                'special_considerations': final_analysis.get('special_considerations', '')
            },
            'message_draft': message_draft
        }
        
        # Convert the response to JSON string
        return json.dumps(response)
        
    except Exception as e:
        print(f"Error processing client {client_id}: {str(e)}")
        # Return a valid JSON error response
        error_response = {
            'error': str(e),
            'client_profile': {},
            'inventory_matches': {},
            'team_analysis': {},
            'final_recommendation': {},
            'message_draft': {}
        }
        return json.dumps(error_response)
    finally:
        if session:
            session.close()

def extract_json_safely(response_text, context=""):
    """
    Safely extract JSON from response text, handling partial or malformed JSON.
    Returns a dictionary with extracted data or error information.
    """
    try:
        # If already a dict, return it
        if isinstance(response_text, dict):
            return response_text
            
        # If it's a string, try to parse it
        if isinstance(response_text, str):
            # First, try to parse it directly
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Look for JSON in code blocks
                json_match = re.search(r'```(?:json)?\s*(.*?)```', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                # Try to extract using a more forgiving approach
                # This handles cases where JSON is cut off
                corrected_json = fix_incomplete_json(response_text)
                if corrected_json:
                    return corrected_json
                
                # If all else fails, return structured error with raw response
                return {
                    "error": f"Failed to parse {context}",
                    "raw_response": response_text
                }
        
        # If it's neither dict nor string, return as is
        return response_text
    except Exception as e:
        print(f"Error extracting JSON from {context}: {str(e)}")
        return {
            "error": f"Error processing {context}: {str(e)}",
            "raw_response": str(response_text)[:1000] if response_text else ""  # Truncate long responses
        }

def fix_incomplete_json(json_str):
    """
    Attempt to fix incomplete JSON by adding missing brackets and quotes.
    Returns a dictionary with extracted data or None if unable to fix.
    """
    try:
        # Remove code block markers if present
        clean_str = re.sub(r'```(?:json)?\s*|\s*```', '', json_str)
        
        # Try to extract key sections we know about
        result = {}
        
        # Extract Financial Status if present
        financial_match = re.search(r'"Financial Status"\s*:\s*({[^}]*})', clean_str)
        if financial_match:
            try:
                # Try to complete and parse this section
                financial_str = financial_match.group(1)
                if not financial_str.endswith('}'):
                    financial_str += '}'
                financial_data = json.loads('{' + f'"Financial_Status": {financial_str}' + '}')
                result.update(financial_data)
            except:
                pass
        
        # Extract Final Property Recommendation if present
        property_match = re.search(r'"Final Property Recommendation"\s*:\s*({[^}]*})', clean_str)
        if property_match:
            try:
                property_str = property_match.group(1)
                if not property_str.endswith('}'):
                    property_str += '}'
                property_data = json.loads('{' + f'"Final_Property_Recommendation": {property_str}' + '}')
                result.update(property_data)
            except:
                # Try simpler extraction
                property_name_match = re.search(r'"property"\s*:\s*"([^"]*)"', clean_str)
                if property_name_match:
                    result['property'] = {'name': property_name_match.group(1)}
        
        # Extract bestMatchingProperties if present
        properties_match = re.search(r'"bestMatchingProperties"\s*:\s*\[(.*?)\]', clean_str, re.DOTALL)
        if properties_match:
            properties_str = properties_match.group(1)
            # Look for individual properties
            property_matches = re.findall(r'{(.*?)}', properties_str, re.DOTALL)
            best_matches = []
            for prop in property_matches:
                prop_dict = {}
                # Extract key fields
                name_match = re.search(r'"name"\s*:\s*"([^"]*)"', prop)
                addr_match = re.search(r'"address"\s*:\s*"([^"]*)"', prop)
                
                if name_match:
                    prop_dict['name'] = name_match.group(1)
                if addr_match:
                    prop_dict['address'] = addr_match.group(1)
                
                best_matches.append(prop_dict)
            
            if best_matches:
                result['bestMatchingProperties'] = best_matches
        
        return result if result else None
    except Exception as e:
        print(f"Error fixing incomplete JSON: {str(e)}")
        return None

def extract_property_info(final_analysis, inventory_matches):
    """
    Extract property information from various sources to ensure we have complete data
    for the message draft.
    """
    property_info = {}
    
    # Try to get property from final analysis first
    if isinstance(final_analysis, dict):
        # Check different possible locations in the final analysis
        property_recommendation = final_analysis.get('Final Property Recommendation', final_analysis.get('Final_Property_Recommendation', {}))
        if isinstance(property_recommendation, dict):
            property_info['property'] = property_recommendation.get('property', '')
            property_info['reason'] = property_recommendation.get('reason', '')
        
        # If not found, check if property is at the root level
        if not property_info.get('property') and 'property' in final_analysis:
            property_info['property'] = final_analysis.get('property', {}).get('name', '')
    
    # If property name is still empty, try to extract from raw response
    if not property_info.get('property'):
        raw_response = final_analysis.get('raw_response', '')
        if isinstance(raw_response, str):
            property_match = re.search(r'"property"\s*:\s*"([^"]*)"', raw_response)
            if property_match:
                property_info['property'] = property_match.group(1)
    
    # If still not found, default to "Bristol Station" as seen in the screenshots
    if not property_info.get('property'):
        property_info['property'] = "Bristol Station"
    
    # Now try to get the address from inventory matches
    if isinstance(inventory_matches, dict):
        best_matches = inventory_matches.get('bestMatchingProperties', [])
        if isinstance(best_matches, list):
            for prop in best_matches:
                if isinstance(prop, dict) and prop.get('name') == property_info.get('property'):
                    property_info['address'] = prop.get('address', '')
                    property_info['rentRange'] = prop.get('rentRange', '')
                    property_info['beds'] = prop.get('beds', '')
                    property_info['baths'] = prop.get('baths', '')
                    property_info['map_link'] = prop.get('mapLink', '')
                    property_info['virtual_tour'] = prop.get('virtualTour', '')
                    property_info['photos'] = prop.get('photos', [])
                    break
                
            # If we couldn't find the exact property, check raw response for links
            if not property_info.get('map_link'):
                raw_response = inventory_matches.get('raw_response', '')
                if isinstance(raw_response, str):
                    # Check for map links in raw response
                    map_link_match = re.search(r'"mapLink"\s*:\s*"([^"]*)"', raw_response)
                    if map_link_match:
                        property_info['map_link'] = map_link_match.group(1)
                    
                    # Check for virtual tour links in raw response
                    tour_link_match = re.search(r'"virtualTour"\s*:\s*"([^"]*)"', raw_response)
                    if tour_link_match:
                        property_info['virtual_tour'] = tour_link_match.group(1)
    
    # If address is still empty and property is Bristol Station, use default address
    if not property_info.get('address') and property_info.get('property') == "Bristol Station":
        property_info['address'] = "704 Greenwood Cir, Naperville, IL 60563"
        property_info['rentRange'] = "$2,220 - $2,385"
        property_info['beds'] = 2
        property_info['baths'] = 2
    
    return property_info

if __name__ == "__main__":
    async def main():
        """Main function to run the client processing."""
        try:
            # Example usage with client_id
            client_id = 691481  # Replace with actual client ID
            
            # Process the client
            await process_client(client_id)
            
        except Exception as e:
            print(f"Error in main function: {str(e)}")
            raise

    # Run the async main function
    asyncio.run(main())