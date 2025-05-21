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
        
        system_message = """You are a Message Draft Expert. Your role is to:
        1. Generate concise and effective message drafts
        2. Adapt tone based on client profile
        3. Highlight key property features
        4. Include special offers and incentives
        
        Focus on:
        - Professional yet friendly tone
        - Clear property highlights
        - Urgency and call to action
        - Special offers and incentives
        - Client-specific preferences
        
        Return your message drafts in a structured format."""
        
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
            draft_request = f"""
            Generate a message draft based on this information:
            
            CLIENT PROFILE:
            {message_data.get('client_profile', '')}
            
            PROPERTY MATCH:
            {message_data.get('property_match', '')}
            
            SPECIAL OFFERS:
            {message_data.get('special_offers', '')}
            
            Please provide:
            1. A concise SMS message (under 160 characters)
            2. A detailed message draft
            3. Key points to highlight
            4. Suggested follow-up questions
            
            Format the response as a JSON object with these categories.
            """
            
            response = self.llm.invoke(draft_request)
            return response.content
            
        except Exception as e:
            return f"Error generating message draft: {str(e)}"

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
        profile_analysis = main_agent.client_profile_agent.analyze_profile(client_profile_data)
        
        # Ensure profile_analysis is valid JSON
        try:
            if isinstance(profile_analysis, str):
                profile_analysis = json.loads(profile_analysis)
        except json.JSONDecodeError:
            profile_analysis = {
                'error': 'Failed to parse profile analysis',
                'raw_response': profile_analysis
            }
        
        # Step 2: Inventory Matching
        inventory_data = {
            'requirements': client_requirements,
            'inventory': client_chat_history,  # Contains inventory information
            'previous_suggestions': previous_messages_text
        }
        inventory_matches = main_agent.inventory_agent.match_properties(inventory_data)
        
        # Ensure inventory_matches is valid JSON
        try:
            if isinstance(inventory_matches, str):
                inventory_matches = json.loads(inventory_matches)
        except json.JSONDecodeError:
            inventory_matches = {
                'error': 'Failed to parse inventory matches',
                'raw_response': inventory_matches
            }
        
        # Step 3: Team Coordination
        analysis_data = {
            'profile_analysis': profile_analysis,
            'inventory_matches': inventory_matches
        }
        final_analysis = main_agent.team_agent.coordinate_analysis(analysis_data)
        
        # Ensure final_analysis is valid JSON
        try:
            if isinstance(final_analysis, str):
                final_analysis = json.loads(final_analysis)
        except json.JSONDecodeError:
            final_analysis = {
                'error': 'Failed to parse final analysis',
                'raw_response': final_analysis
            }
        
        # Create the final response structure
        response = {
            'client_profile': profile_analysis,
            'inventory_matches': inventory_matches,
            'team_analysis': final_analysis,
            'final_recommendation': {
                'selected_property': final_analysis.get('final_property_recommendation', {}),
                'match_justification': final_analysis.get('match_justification', ''),
                'special_considerations': final_analysis.get('special_considerations', '')
            }
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
            'final_recommendation': {}
        }
        return json.dumps(error_response)
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