import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import re
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

# Get environment variables
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Validate required environment variables
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Please set it in your .env file.")

# Ensure DATABASE_URL starts with postgresql://
if not DATABASE_URL.startswith('postgresql://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

def sendDiscordSanityCheckNoteAlert(textContent, channel_id='1379432907130802306'):
    """Send message to Discord channel with error handling."""
    try:
        # Clean and prepare content for Discord
        content = str(textContent)
        
        # Remove or escape problematic characters that can cause 400 errors
        content = content.replace('```json\n{', '```json\n{\n  ')
        content = content.replace('\\"', '"')  # Fix escaped quotes
        
        # Ensure content is within Discord's limits
        if len(content) > 2000:
            content = content[:1950] + "\n... (truncated)"
        
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        payload = {
            "content": content
        }
        headers = {
            "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            print(f"✅ Discord message sent successfully")
        elif response.status_code == 400:
            print(f"⚠️ Discord API 400 error - Content issue detected")
            print(f"Content length: {len(content)}")
            print(f"First 200 chars: {content[:200]}")
            # Try sending a simplified version
            simple_content = "🚨 **Analysis Complete** - Content formatting issue prevented full display"
            simple_payload = {"content": simple_content}
            fallback_response = requests.post(url, headers=headers, json=simple_payload)
            if fallback_response.status_code == 200:
                print(f"✅ Sent simplified Discord message as fallback")
        else:
            print(f"⚠️ Discord API responded with status {response.status_code}")
            if response.text:
                print(f"Response: {response.text}")
        return response
    except Exception as e:
        print(f"❌ Error sending Discord message: {e}")
        return None

class InventoryAgent:
    def __init__(self, show_thinking: bool = True, send_to_discord: bool = False, discord_channel_id: str = '1379432907130802306', verbose: bool = True, silent_mode: bool = False):
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
        self.geocoder = Nominatim(user_agent="inventory_agent")
        self.session_factory = sessionmaker(bind=create_engine(DATABASE_URL))
        self.show_thinking = show_thinking
        self.send_to_discord = send_to_discord
        self.discord_channel_id = discord_channel_id
        self.verbose = verbose
        self.silent_mode = silent_mode  # When True, suppresses stdout but still logs to Discord
        self.verbose_logs = []  # Capture all verbose logs for Discord
        
        if self.verbose and not self.silent_mode:
            print("🔧 Initializing InventoryAgent with verbose logging enabled")
            print(f"   📡 Model: {GENAI_MODEL}")
            print(f"   🎯 Temperature: 0.3")
            print(f"   📺 Show thinking: {show_thinking}")
            print(f"   📨 Discord enabled: {send_to_discord}")
            print(f"   🔇 Silent mode: {silent_mode}")
        
        self.verbose_log("🔧 InventoryAgent initialized", "INFO")
        self.verbose_log(f"Model: {GENAI_MODEL}, Temperature: 0.3", "DEBUG")
        
        # Define the main prompt template with thinking process
        self.prompt_template = PromptTemplate(
            input_variables=["client_info", "client_requirements", "inventory_data", "previous_messages"],
            template="""
            You are an expert property matching agent specializing in neighborhood-based recommendations.
            
            TASK: Find the best property match for the client based on their neighborhood preferences and proximity.
            
            CLIENT INFORMATION:
            {client_info}
            
            CLIENT REQUIREMENTS:
            {client_requirements}
            
            AVAILABLE INVENTORY:
            {inventory_data}
            
            PREVIOUS MESSAGES TO AVOID DUPLICATES:
            {previous_messages}
            
            ANALYSIS FRAMEWORK:
            You must think through this systematically and show your reasoning process.
            
            STEP 1 - CLIENT PROFILE ANALYSIS:
            - Analyze client's current location and preferred neighborhoods
            - Assess financial capacity and timeline urgency
            - Identify special requirements and deal-breakers
            - Evaluate approval likelihood and risk factors
            - Provide detailed reasoning for client assessment
            
            STEP 2 - INVENTORY EVALUATION:
            - List all available properties with key details
            - Calculate neighborhood proximity for each property
            - Filter by budget compatibility
            - Check unit specifications match
            - Calculate net effective rent for properties with move-in specials (assume 12-month lease)
            - Identify special offers and commission opportunities
            - Provide specific analysis for each property
            
            STEP 3 - PROPERTY RANKING:
            - Score each property on proximity to preferred neighborhoods
            - Evaluate budget fit and value proposition (including net effective rent)
            - Consider move-in timeline alignment
            - Factor in special requirements (parking, pets, etc.)
            - Check against previous suggestions to avoid duplicates
            - Show clear ranking methodology
            
            STEP 4 - FINAL SELECTION:
            - Select the best match with detailed reasoning
            - Explain why this property beats other options
            - Highlight key selling points for the client
            - Show net effective rent if applicable
            - Craft persuasive but concise SMS message
            - Provide specific justification for the choice
            
            RESPONSE FORMAT:
            Return a JSON object with this exact structure:
            {{
                "thinking_process": {{
                    "client_analysis": "[Analyze the client's profile including location preferences, budget constraints, timeline, and special needs. Explain their current situation and what they're looking for. Include specific details about their requirements and any constraints.]",
                    "inventory_evaluation": "[Evaluate each available property individually, discussing proximity to client's preferred areas, budget fit, and how well it matches their requirements. Include specific property details, addresses, base rent, move-in specials, net effective rent calculations, and neighborhood analysis.]",
                    "property_ranking": "[Explain how you scored and ranked the properties. Detail the methodology used and why certain properties ranked higher than others. Show specific scoring criteria and calculations.]",
                    "selection_reasoning": "[Provide specific reasons why the selected property is the best choice. Compare it to alternatives and explain what makes it superior for this particular client. Include proximity analysis, budget considerations, and feature matching.]",
                    "duplicate_check": "[Confirm that the selected property hasn't been suggested to this client before by checking previous messages. List previous suggestions and verify the new selection is different.]"
                }},
                "property_matches": [
                    {{
                        "address": "[Property address]",
                        "neighborhood": "[Neighborhood name]",
                        "base_rent": "[Original monthly rent range]",
                        "move_in_special": "[Any move-in incentives like '1 month free', '$500 off', etc.]",
                        "net_effective_rent": "[Calculated effective rent with specials - use base_rent if no specials]",
                        "specs": "[Bedroom/bathroom count]",
                        "proximity_score": "[Distance/proximity to client preferences]",
                        "budget_fit": "[How it fits client budget including net effective consideration]",
                        "pros": "[Key advantages]",
                        "cons": "[Potential drawbacks]",
                        "ranking": "[1-5, where 1 is best match]"
                    }}
                ],
                "selected_property": {{
                    "address": "[Selected property address]",
                    "neighborhood": "[Neighborhood]",
                    "base_rent": "[Original monthly rent range]",
                    "net_effective_rent": "[Effective rent with specials calculated]",
                    "move_in_date": "[Available move-in date]",
                    "key_features": "[Main selling points]",
                    "proximity_advantage": "[Why location works for client]",
                    "special_offers": "[Any incentives or deals]",
                    "savings_details": "[Breakdown of cost savings from specials]"
                }},
                "message": "Ok good news [client_name], I found a great option at [property_name]  ([address]). You can enjoy write here the average calculated on savings [special_offers],  (only write this message if a special is found in [special_offers] and [savings_details]). Let me know when you’d like to schedule a showing!"
            }}
            
            IMPORTANT GUIDELINES:
            - Provide substantial content for ALL thinking_process fields
            - Calculate net effective rent: (Base Rent * 12 - Move-in Specials Value) ÷ 12
            - Use specific property data from the inventory
            - Reference actual client requirements and preferences
            - Give concrete reasons for your decisions
            - Avoid generic responses like "No analysis available"
            - Include specific addresses, prices, and neighborhood details
            - Show your work and reasoning at each step
            - Be thorough and comprehensive in your analysis
            - Format the final message with actual property name, address, and URL-encoded Google Maps link
            - Include the client's actual name in the greeting
            - Use net effective rent when move-in specials apply, otherwise use base rent
            
            NET EFFECTIVE RENT CALCULATION:
            - For properties with move-in specials (free months, cash back, etc.)
            - Formula: (Monthly Rent × 12 - Total Move-in Savings) ÷ 12
            - Example: $2000/month with 1 month free = ($2000×12 - $2000) ÷ 12 = $1833 net effective
            - Example: $1800/month with $500 off = ($1800×12 - $500) ÷ 12 = $1758 net effective
            - Always show both base rent and net effective rent when specials apply
            - If no specials, net_effective_rent should equal base_rent
            
            MATCHING CRITERIA PRIORITY:
            1. NEIGHBORHOOD PROXIMITY (40%) - Distance to preferred neighborhoods
            2. BUDGET COMPATIBILITY (25%) - Rent within client's range
            3. UNIT SPECIFICATIONS (20%) - Bedroom/bathroom requirements
            4. TIMELINE ALIGNMENT (10%) - Move-in date compatibility
            5. SPECIAL REQUIREMENTS (5%) - Parking, pets, amenities
            
            MESSAGE FORMAT REQUIREMENTS:
            - Start with "Ok good news [client_name], I found a great option at"
            - Include property name and only full address in parentheses
            - Mention the special offer and savings (use the fields "special_offers" and "savings_details")
            - If there is a special offer, compute the average dollar savings from the numeric values in [savings_details].  
              (For example, if savings_details="With a 14-16 month lease, Mark could save between $499.75 and $607 per month", calculate the midpoint: round((499.75 + 607) / 2) → 553. Then round up or down to a nice number, e.g. $600.)  
            - Instead of writing a range (“$499.75–$607”), say “averaging about $600 saved on a 16-month lease.”  
            - Use a comma before “saving…” rather than an em-dash.  
            - Do NOT include a Google Maps link or any net-effective‐rent calculation
            - End with a friendly “Let me know when you’d like to schedule a showing!”
            - Keep professional, friendly tone and concise message
            - Use actual data from selected property
            
            Focus on location proximity as the primary matching factor.
            Ensure all thinking process sections contain meaningful analysis and reasoning.
            """
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def verbose_log(self, message: str, level: str = "INFO"):
        """Enhanced logging for verbose mode with Discord capture."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Always capture logs for Discord if verbose is enabled
        if self.verbose:
            log_entry = f"[{timestamp}] {level}: {message}"
            self.verbose_logs.append(log_entry)
        
        # Only print to stdout if not in silent mode
        if self.verbose and not self.silent_mode:
            if level == "DEBUG":
                print(f"🔍 [{timestamp}] DEBUG: {message}")
            elif level == "INFO":
                print(f"ℹ️  [{timestamp}] INFO: {message}")
            elif level == "SUCCESS":
                print(f"✅ [{timestamp}] SUCCESS: {message}")
            elif level == "WARNING":
                print(f"⚠️  [{timestamp}] WARNING: {message}")
            elif level == "ERROR":
                print(f"❌ [{timestamp}] ERROR: {message}")
            else:
                print(f"📝 [{timestamp}] {message}")

    def get_verbose_logs_summary(self) -> str:
        """Get formatted summary of all verbose logs for Discord."""
        if not self.verbose_logs:
            return "No verbose logs captured."
        
        summary = "** PROCESSING LOGS:**\n```\n"
        # Get last 50 logs to avoid Discord message limits
        recent_logs = self.verbose_logs[-50:] if len(self.verbose_logs) > 50 else self.verbose_logs
        summary += "\n".join(recent_logs)
        summary += "\n```"
        
        return summary

    def get_db_session(self):
        """Create and return a database session."""
        self.verbose_log("Creating database session")
        return self.session_factory()

    def get_client_data(self, client_id: int) -> Dict:
        """Fetch comprehensive client data from database."""
        self.verbose_log(f"Fetching client data for ID: {client_id}")
        session = self.get_db_session()
        try:
            # Get client basic info
            self.verbose_log("Querying client basic information")
            client_query = text("""
                SELECT id, fullname, email, addresses, neighborhood, max_salary
                FROM public.client 
                WHERE id = :client_id
            """)
            client_result = session.execute(client_query, {"client_id": client_id}).fetchone()
            
            if not client_result:
                self.verbose_log(f"No client found with ID: {client_id}", "ERROR")
                raise ValueError(f"No client found with ID: {client_id}")
            
            client_dict = dict(client_result._mapping)
            self.verbose_log(f"Found client: {client_dict.get('fullname', 'Unknown')}")
            
            # Get client requirements
            self.verbose_log("Querying client requirements")
            req_query = text("""
                SELECT budget, budget_max, move_in_date, move_in_date_max,
                       beds, baths, parking, pets, washer_dryer,
                       neighborhood, zip, sqft, sqft_max,
                       building_must_haves, unit_must_haves,
                       special_needs, preference, comment,
                       lease_term, section8, monthly_income, credit_score, amenities
                FROM requirements 
                WHERE client_id = :client_id
            """)
            req_result = session.execute(req_query, {"client_id": client_id}).fetchone()
            
            req_dict = dict(req_result._mapping) if req_result else {}
            self.verbose_log(f"Requirements found: Budget ${req_dict.get('budget', 'N/A')}-${req_dict.get('budget_max', 'N/A')}")
            
            # Get inventory/notes data
            self.verbose_log("Querying inventory data")
            inventory_query = text("""
                SELECT body 
                FROM note 
                WHERE client_id = :client_id 
                AND subject = 'Inventory Discovery For Client'
                ORDER BY created DESC
            """)
            inventory_results = session.execute(inventory_query, {"client_id": client_id}).fetchall()
            self.verbose_log(f"Found {len(inventory_results)} inventory notes")
            
            # Get previous messages
            self.verbose_log("Querying previous messages")
            messages_query = text("""
                SELECT message, created
                FROM textmessage 
                WHERE client_id = :client_id 
                AND is_incoming = false
                ORDER BY created DESC
                LIMIT 10
            """)
            message_results = session.execute(messages_query, {"client_id": client_id}).fetchall()
            self.verbose_log(f"Found {len(message_results)} previous messages")
            
            return {
                "client": client_dict,
                "requirements": req_dict,
                "inventory": [row[0] for row in inventory_results],
                "previous_messages": [(row[0], row[1]) for row in message_results]
            }
            
        finally:
            self.verbose_log("Closing database session")
            session.close()

    def format_client_info(self, client_data: Dict) -> str:
        """Format client information for the prompt."""
        client = client_data.get("client", {})
        formatted_info = f"""
        Name: {client.get('fullname', 'N/A')}
        Current Address: {client.get('addresses', 'N/A')}
        Preferred Neighborhood: {client.get('neighborhood', 'N/A')}
        Email: {client.get('email', 'N/A')}
        Max Salary: ${client.get('max_salary', 'N/A')}
        """
        self.verbose_log("Formatted client information")
        return formatted_info

    def format_requirements(self, requirements: Dict) -> str:
        """Format client requirements for the prompt."""
        self.verbose_log("Formatting client requirements")
        req_parts = []
        
        # Budget
        if requirements.get('budget') or requirements.get('budget_max'):
            budget_range = f"${requirements.get('budget', '')}-${requirements.get('budget_max', '')}"
            req_parts.append(f"Budget: {budget_range}")
            self.verbose_log(f"Budget range: {budget_range}")
        
        # Move-in dates
        if requirements.get('move_in_date'):
            move_in = f"{requirements['move_in_date']}"
            if requirements.get('move_in_date_max'):
                move_in += f" to {requirements['move_in_date_max']}"
            req_parts.append(f"Move-in: {move_in}")
            self.verbose_log(f"Move-in timeline: {move_in}")
        
        # Unit specs
        if requirements.get('beds'):
            req_parts.append(f"Bedrooms: {requirements['beds']}")
            self.verbose_log(f"Bedroom requirement: {requirements['beds']}")
        if requirements.get('baths'):
            req_parts.append(f"Bathrooms: {requirements['baths']}")
            self.verbose_log(f"Bathroom requirement: {requirements['baths']}")
        
        # Location preferences
        if requirements.get('neighborhood'):
            neighborhoods = ', '.join(requirements['neighborhood']) if isinstance(requirements['neighborhood'], list) else requirements['neighborhood']
            req_parts.append(f"Preferred Neighborhoods: {neighborhoods}")
            self.verbose_log(f"Preferred neighborhoods: {neighborhoods}")
        
        if requirements.get('zip'):
            zip_codes = ', '.join(requirements['zip']) if isinstance(requirements['zip'], list) else requirements['zip']
            req_parts.append(f"ZIP Codes: {zip_codes}")
            self.verbose_log(f"ZIP codes: {zip_codes}")
        
        # Additional requirements
        if requirements.get('parking'):
            req_parts.append(f"Parking: {requirements['parking']}")
        if requirements.get('pets'):
            req_parts.append(f"Pets: {requirements['pets']}")
        if requirements.get('special_needs'):
            req_parts.append(f"Special Needs: {requirements['special_needs']}")
        if requirements.get('amenities'):
            amenities = ', '.join(requirements['amenities']) if isinstance(requirements['amenities'], list) else requirements['amenities']
            req_parts.append(f"Amenities: {amenities}")
        
        self.verbose_log(f"Formatted {len(req_parts)} requirement categories")
        return "\n".join(req_parts)

    def format_inventory(self, inventory_list: List[str]) -> str:
        """Format inventory data for the prompt."""
        if not inventory_list:
            self.verbose_log("No inventory data available", "WARNING")
            return "No inventory data available."
        
        # Combine all inventory notes
        all_inventory = "\n\n".join(inventory_list)
        self.verbose_log(f"Formatted inventory data: {len(all_inventory)} characters")
        
        # Clean up and format
        return f"AVAILABLE PROPERTIES:\n{all_inventory}"

    def format_previous_messages(self, messages: List[Tuple]) -> str:
        """Format previous messages to avoid duplicates."""
        if not messages:
            self.verbose_log("No previous messages found")
            return "No previous messages."
        
        formatted_messages = []
        for message, date in messages:
            formatted_messages.append(f"[{date}] {message}")
        
        self.verbose_log(f"Formatted {len(messages)} previous messages")
        return "\n".join(formatted_messages)

    def format_discord_message(self, analysis: Dict, client_data: Dict) -> str:
        """Format the comprehensive analysis for Discord with all verbose logs."""
        client = client_data.get("client", {})
        client_name = client.get('fullname', 'Unknown Client')
        client_id = client.get('id', 'N/A')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Construct FUB link
        fub_link = f"https://app.followupboss.com/2/people/view/{client_id}" if client_id != 'N/A' else "FUB link not available"
        
        # Start building the comprehensive Discord message
        discord_msg = f"""🏠 **COMPREHENSIVE PROPERTY ANALYSIS REPORT** 🏠
📅 **Timestamp:** {timestamp}
👤 **Client:** {client_name} (ID: {client_id})
🔗 **FUB Link:** {fub_link}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{self.get_verbose_logs_summary()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 **ANALYSIS SUMMARY**
"""
        
        thinking = analysis.get("thinking_process", {})
        
        # Show brief summaries instead of full content
        sections = [
            ("📊 **Client Analysis:**", "client_analysis"),
            ("🏠 **Inventory Evaluation:**", "inventory_evaluation"),
            ("📈 **Property Ranking:**", "property_ranking"),
            ("🎯 **Selection Reasoning:**", "selection_reasoning"),
            ("✅ **Duplicate Check:**", "duplicate_check")
        ]
        
        for title, key in sections:
            content = thinking.get(key, "Content not provided")
            # Show only first 200 characters as summary
            summary = content[:200] + "..." if len(content) > 200 else content
            discord_msg += f"""
{title} {summary}

"""
        
        discord_msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # Property matches with detailed analysis
        matches = analysis.get("property_matches", [])
        if matches:
            discord_msg += f"\n🏘️ **PROPERTY MATCHES ANALYSIS ({len(matches)} found):**\n"
            for i, prop in enumerate(matches[:3], 1):
                base_rent = prop.get('base_rent', prop.get('price', 'N/A'))
                net_effective = prop.get('net_effective_rent', 'N/A')
                move_in_special = prop.get('move_in_special', 'None')
                
                discord_msg += f"""
**Option {i}: {prop.get('address', 'Unknown Address')}**
🏘️ Neighborhood: {prop.get('neighborhood', 'N/A')}
💰 Base Rent: {base_rent}
"""
                # Show net effective rent if different from base rent
                if net_effective and net_effective != 'N/A' and net_effective != base_rent:
                    discord_msg += f"💡 Net Effective: {net_effective}\n"
                if move_in_special and move_in_special != 'None':
                    discord_msg += f"🎁 Special: {move_in_special}\n"
                
                discord_msg += f"""🏠 Specs: {prop.get('specs', 'N/A')}
📍 Proximity: {prop.get('proximity_score', 'N/A')}
💵 Budget Fit: {prop.get('budget_fit', 'N/A')}
🏆 Ranking: #{prop.get('ranking', 'N/A')}
✅ Pros: {prop.get('pros', 'N/A')}
"""
                if prop.get('cons'):
                    discord_msg += f"❌ Cons: {prop.get('cons')}\n"
                discord_msg += "\n"
        
        # Selected property with comprehensive details
        selected = analysis.get("selected_property", {})
        if selected:
            base_rent = selected.get('base_rent', selected.get('price', 'N/A'))
            net_effective = selected.get('net_effective_rent', 'N/A')
            savings_details = selected.get('savings_details', 'N/A')
            
            discord_msg += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **FINAL SELECTED PROPERTY:**
🏠 **Address:** {selected.get('address', 'N/A')}
🏘️ **Neighborhood:** {selected.get('neighborhood', 'N/A')}
💰 **Base Rent:** {base_rent}
"""
            # Show net effective rent if different from base rent
            if net_effective and net_effective != 'N/A' and net_effective != base_rent:
                discord_msg += f"💡 **Net Effective:** {net_effective}\n"
            if savings_details and savings_details != 'N/A':
                discord_msg += f"💎 **Savings:** {savings_details}\n"
                
            discord_msg += f"""📅 **Move-in:** {selected.get('move_in_date', 'N/A')}
⭐ **Key Features:** {selected.get('key_features', 'N/A')}
📍 **Location Advantage:** {selected.get('proximity_advantage', 'N/A')}
"""
            if selected.get('special_offers'):
                discord_msg += f"🎁 **Special Offers:** {selected.get('special_offers')}\n"
        
        # Final SMS message only
        final_message = analysis.get("message", "No message generated")
        discord_msg += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱 **CLIENT COMMUNICATION:**
> {final_message}
"""
        
        return discord_msg

    def format_complete_analysis_for_discord(self, analysis: Dict) -> str:
        """Format the complete JSON analysis result for Discord."""
        try:
            # Safely format JSON for Discord
            json_str = json.dumps(analysis, indent=2, ensure_ascii=False)
            
            # Show complete thinking process sections instead of character counts
            thinking = analysis.get("thinking_process", {})
            matches = analysis.get("property_matches", [])
            selected = analysis.get("selected_property", {})
            message = analysis.get("message", "")
            
            complete_analysis = f"""📋 **COMPLETE TECHNICAL ANALYSIS:**

🧠 **COMPLETE THINKING PROCESS:**

**CLIENT ANALYSIS:**
```
{thinking.get('client_analysis', 'Not provided')[:1200]}
```

**INVENTORY EVALUATION:**
```
{thinking.get('inventory_evaluation', 'Not provided')[:1200]}
```

**PROPERTY RANKING:**
```
{thinking.get('property_ranking', 'Not provided')[:1200]}
```

**SELECTION REASONING:**
```
{thinking.get('selection_reasoning', 'Not provided')[:1200]}
```

**DUPLICATE CHECK:**
```
{thinking.get('duplicate_check', 'Not provided')[:800]}
```

📊 **ANALYSIS SUMMARY:**
• Properties analyzed: {len(matches)}
• Selected property: {selected.get('address', 'Not specified')}
• Price range: {selected.get('price', 'Not specified')}
• Final message: {message[:150]}{'...' if len(message) > 150 else ''}

**Full Analysis Size:** {len(json_str)} characters"""
            
            return complete_analysis
            
        except Exception as e:
            return f"📋 **ANALYSIS RESULT:** Error formatting analysis - {str(e)}"

    def send_comprehensive_discord_analysis(self, analysis: Dict, client_data: Dict):
        """Send comprehensive analysis to Discord in optimized chunks."""
        if not self.send_to_discord:
            return
            
        self.verbose_log("Preparing comprehensive Discord analysis", "INFO")
        
        # Get client info for FUB link
        client = client_data.get("client", {})
        client_id = client.get('id', 'N/A')
        fub_link = f"https://app.followupboss.com/2/people/view/{client_id}" if client_id != 'N/A' else "FUB link not available"
        
        # Send the main analysis report
        discord_message = self.format_discord_message(analysis, client_data)
        
        # Send the complete JSON analysis result
        complete_analysis_msg = self.format_complete_analysis_for_discord(analysis)
        
        # Format the complete JSON structure for Discord
        complete_json = json.dumps(analysis, indent=2, ensure_ascii=False)
        json_message = f"""📋 **COMPLETE JSON ANALYSIS RESULT:**
```json
{complete_json[:1800]}{'...' if len(complete_json) > 1800 else ''}
```

**Full JSON Size:** {len(complete_json)} characters
**Matches Console Output:** ✅"""
        
        # Discord has a 2000 character limit, so we need to split intelligently
        max_chunk_size = 1900  # Leave buffer for safety
        
        # Send main analysis - split by sections if too long
        if len(discord_message) <= max_chunk_size:
            sendDiscordSanityCheckNoteAlert(discord_message, self.discord_channel_id)
            self.verbose_log("Sent main analysis to Discord", "SUCCESS")
        else:
            # Split main analysis by thinking sections to avoid duplicates
            parts = discord_message.split("🧠 **DETAILED AGENT THINKING PROCESS**")
            
            # Send header part
            if len(parts) > 0:
                header = parts[0] + "🧠 **DETAILED AGENT THINKING PROCESS**"
                sendDiscordSanityCheckNoteAlert(header, self.discord_channel_id)
            
            # Send thinking sections
            if len(parts) > 1:
                thinking_content = parts[1]
                
                # Split thinking content by sections
                thinking_sections = thinking_content.split("**Client Analysis:**")
                if len(thinking_sections) > 1:
                    sections_to_send = thinking_sections[1].split("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    
                    current_chunk = "**Client Analysis:**"
                    for section in sections_to_send:
                        if len(current_chunk + section) <= max_chunk_size:
                            current_chunk += section
                        else:
                            if current_chunk:
                                sendDiscordSanityCheckNoteAlert(current_chunk, self.discord_channel_id)
                            current_chunk = section
                    
                    # Send final chunk
                    if current_chunk:
                        sendDiscordSanityCheckNoteAlert(current_chunk, self.discord_channel_id)
            
            self.verbose_log("Sent main analysis in multiple parts", "SUCCESS")
        
        # Send complete technical analysis as separate message
        if len(complete_analysis_msg) <= max_chunk_size:
            sendDiscordSanityCheckNoteAlert(complete_analysis_msg, self.discord_channel_id)
            self.verbose_log("Sent complete technical analysis", "SUCCESS")
        else:
            # Split complete analysis by sections
            sections = complete_analysis_msg.split("**CLIENT ANALYSIS:**")
            sendDiscordSanityCheckNoteAlert(sections[0], self.discord_channel_id)
            
            if len(sections) > 1:
                remaining_content = "**CLIENT ANALYSIS:**" + sections[1]
                content_sections = remaining_content.split("**INVENTORY EVALUATION:**")
                
                for i, section in enumerate(content_sections):
                    if i == 0:
                        chunk = section
                    else:
                        chunk = "**INVENTORY EVALUATION:**" + section
                    
                    if len(chunk) <= max_chunk_size:
                        sendDiscordSanityCheckNoteAlert(chunk, self.discord_channel_id)
                    else:
                        # Further split if still too long
                        lines = chunk.split('\n')
                        current_chunk = ""
                        for line in lines:
                            if len(current_chunk + line + '\n') <= max_chunk_size:
                                current_chunk += line + '\n'
                            else:
                                if current_chunk:
                                    sendDiscordSanityCheckNoteAlert(current_chunk, self.discord_channel_id)
                                current_chunk = line + '\n'
                        if current_chunk:
                            sendDiscordSanityCheckNoteAlert(current_chunk, self.discord_channel_id)
            
            self.verbose_log("Sent complete technical analysis in parts", "SUCCESS")
        
        # Send complete JSON structure
        if len(json_message) <= max_chunk_size:
            sendDiscordSanityCheckNoteAlert(json_message, self.discord_channel_id)
            self.verbose_log("Sent complete JSON structure", "SUCCESS")
        else:
            # Split JSON into multiple parts if too long
            json_parts = []
            
            # Try to split by major sections in the JSON
            if '"thinking_process"' in complete_json:
                # Split thinking process
                thinking_part = complete_json.split('"thinking_process"')[1].split('"property_matches"')[0]
                json_parts.append(f"""📋 **JSON PART 1 - THINKING PROCESS:**
```json
"thinking_process": {thinking_part}
```""")
                
                # Property matches part
                if '"property_matches"' in complete_json:
                    matches_part = complete_json.split('"property_matches"')[1].split('"selected_property"')[0]
                    json_parts.append(f"""📋 **JSON PART 2 - PROPERTY MATCHES:**
```json
"property_matches": {matches_part}
```""")
                
                # Selected property and message
                if '"selected_property"' in complete_json:
                    remaining_part = complete_json.split('"selected_property"')[1]
                    json_parts.append(f"""📋 **JSON PART 3 - SELECTED PROPERTY & MESSAGE:**
```json
"selected_property": {remaining_part}
```""")
            else:
                # Fallback: split by character chunks
                chunk_size = 1500
                for i in range(0, len(complete_json), chunk_size):
                    chunk = complete_json[i:i+chunk_size]
                    json_parts.append(f"""📋 **JSON PART {len(json_parts)+1}:**
```json
{chunk}
```""")
            
            # Send all JSON parts
            for i, part in enumerate(json_parts, 1):
                if len(part) <= max_chunk_size:
                    sendDiscordSanityCheckNoteAlert(part, self.discord_channel_id)
                else:
                    # Further split if still too long
                    lines = part.split('\n')
                    current_chunk = ""
                    for line in lines:
                        if len(current_chunk + line + '\n') <= max_chunk_size:
                            current_chunk += line + '\n'
                        else:
                            if current_chunk:
                                sendDiscordSanityCheckNoteAlert(current_chunk.rstrip(), self.discord_channel_id)
                            current_chunk = line + '\n'
                    if current_chunk:
                        sendDiscordSanityCheckNoteAlert(current_chunk.rstrip(), self.discord_channel_id)
            
            self.verbose_log(f"Sent complete JSON in {len(json_parts)} parts", "SUCCESS")

    def print_thinking_process(self, analysis: Dict):
        """Print the agent's thinking process in a readable format with enhanced verbosity."""
        if not self.show_thinking:
            return
            
        print("\n" + "="*80)
        print("🧠 DETAILED AGENT THINKING PROCESS")
        print("="*80)
        
        thinking = analysis.get("thinking_process", {})
        
        # Enhanced thinking display with character counts and quality assessment
        sections = [
            ("📊 CLIENT ANALYSIS", "client_analysis"),
            ("🏠 INVENTORY EVALUATION", "inventory_evaluation"),
            ("📈 PROPERTY RANKING", "property_ranking"),
            ("🎯 SELECTION REASONING", "selection_reasoning"),
            ("✅ DUPLICATE CHECK", "duplicate_check")
        ]
        
        for title, key in sections:
            content = thinking.get(key, "Content not provided")
            print(f"\n{title}:")
            print("-" * 60)
            print(content)
            
            if self.verbose:
                print(f"\n📏 Analysis: {len(content)} characters")
                if len(content) < 100:
                    print("⚠️  Warning: Content may be too brief for thorough analysis")
                elif len(content) > 1000:
                    print("✅ Good: Comprehensive analysis provided")
                else:
                    print("ℹ️  Moderate length analysis")
        
        # Show property matches if available
        matches = analysis.get("property_matches", [])
        if matches:
            print(f"\n🏘️ AVAILABLE PROPERTY MATCHES ({len(matches)} found):")
            print("-" * 60)
            for i, prop in enumerate(matches, 1):
                print(f"\n{i}. {prop.get('address', 'Unknown Address')}")
                print(f"   🏘️ Neighborhood: {prop.get('neighborhood', 'N/A')}")
                print(f"   💰 Price: {prop.get('price', 'N/A')}")
                print(f"   🏠 Specs: {prop.get('specs', 'N/A')}")
                print(f"   📍 Proximity: {prop.get('proximity_score', 'N/A')}")
                print(f"   💵 Budget Fit: {prop.get('budget_fit', 'N/A')}")
                print(f"   🏆 Ranking: #{prop.get('ranking', 'N/A')}")
                print(f"   ✅ Pros: {prop.get('pros', 'N/A')}")
                if prop.get('cons'):
                    print(f"   ❌ Cons: {prop.get('cons')}")
                    
                if self.verbose:
                    print(f"   📊 Data Quality:")
                    print(f"      - Address: {'✅' if prop.get('address') else '❌'}")
                    print(f"      - Price: {'✅' if prop.get('price') else '❌'}")
                    print(f"      - Ranking: {'✅' if prop.get('ranking') else '❌'}")
        
        # Show selected property
        selected = analysis.get("selected_property", {})
        if selected:
            print(f"\n🎯 FINAL SELECTED PROPERTY:")
            print("-" * 60)
            print(f"🏠 Address: {selected.get('address', 'N/A')}")
            print(f"🏘️ Neighborhood: {selected.get('neighborhood', 'N/A')}")
            print(f"💰 Price: {selected.get('price', 'N/A')}")
            print(f"📅 Move-in: {selected.get('move_in_date', 'N/A')}")
            print(f"⭐ Key Features: {selected.get('key_features', 'N/A')}")
            print(f"📍 Location Advantage: {selected.get('proximity_advantage', 'N/A')}")
            if selected.get('special_offers'):
                print(f"🎁 Special Offers: {selected.get('special_offers')}")
                
            if self.verbose:
                print(f"\n📊 Selection Quality Assessment:")
                quality_score = 0
                total_fields = 0
                for field in ['address', 'neighborhood', 'price', 'key_features', 'proximity_advantage']:
                    total_fields += 1
                    if selected.get(field) and selected.get(field) != 'N/A':
                        quality_score += 1
                        print(f"   ✅ {field.replace('_', ' ').title()}: Complete")
                    else:
                        print(f"   ❌ {field.replace('_', ' ').title()}: Missing")
                
                percentage = (quality_score / total_fields) * 100
                print(f"   📈 Overall Completeness: {percentage:.1f}% ({quality_score}/{total_fields})")
        
        print("\n" + "="*80)

    async def find_property_match(self, client_id: int) -> str:
        """Main method to find property matches for a client."""
        try:
            self.verbose_log(f"🔍 STARTING PROPERTY SEARCH FOR CLIENT ID: {client_id}", "INFO")
            self.verbose_log("=" * 80)
            
            # Get all client data
            self.verbose_log("📊 PHASE 1: DATA COLLECTION", "INFO")
            client_data = self.get_client_data(client_id)
            self.verbose_log("✅ Client data retrieved successfully", "SUCCESS")
            
            # Format data for the prompt
            self.verbose_log("📋 PHASE 2: DATA FORMATTING", "INFO")
            client_info = self.format_client_info(client_data)
            self.verbose_log("Formatted client information")
            
            requirements = self.format_requirements(client_data.get("requirements", {}))
            self.verbose_log("Formatted client requirements")
            
            inventory = self.format_inventory(client_data.get("inventory", []))
            self.verbose_log("Formatted inventory data")
            
            previous_messages = self.format_previous_messages(client_data.get("previous_messages", []))
            self.verbose_log("Formatted previous messages")
            
            self.verbose_log("🧠 PHASE 3: AI ANALYSIS", "INFO")
            self.verbose_log("Sending request to AI model for analysis...")
            
            if self.verbose and not self.silent_mode:
                print("\n" + "="*80)
                print("📝 PROMPT DATA BEING SENT TO AI:")
                print("="*80)
                print(f"CLIENT INFO:\n{client_info}")
                print(f"\nREQUIREMENTS:\n{requirements}")
                print(f"\nINVENTORY:\n{inventory[:500]}..." if len(inventory) > 500 else f"\nINVENTORY:\n{inventory}")
                print(f"\nPREVIOUS MESSAGES:\n{previous_messages}")
                print("="*80)
            
            # Log prompt data for Discord
            self.verbose_log(f"CLIENT INFO: {client_info[:200]}...", "DEBUG")
            self.verbose_log(f"REQUIREMENTS: {requirements[:200]}...", "DEBUG")
            self.verbose_log(f"INVENTORY: {len(inventory)} characters of inventory data", "DEBUG")
            
            # Run the LangChain
            self.verbose_log("🤖 Executing AI chain...", "INFO")
            response = await self.chain.arun(
                client_info=client_info,
                client_requirements=requirements,
                inventory_data=inventory,
                previous_messages=previous_messages
            )
            
            self.verbose_log("✅ AI response received", "SUCCESS")
            self.verbose_log(f"Response length: {len(response)} characters")
            
            if self.verbose and not self.silent_mode:
                print("\n" + "="*80)
                print("🤖 RAW AI RESPONSE:")
                print("="*80)
                print(response)
                print("="*80)
            
            # Log response summary for Discord
            self.verbose_log(f"RAW AI RESPONSE: {response[:300]}...", "DEBUG")
            
            # Clean and validate JSON response
            self.verbose_log("🔧 PHASE 4: RESPONSE PROCESSING", "INFO")
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
                self.verbose_log("Removed JSON markdown prefix")
            if response.endswith('```'):
                response = response[:-3]
                self.verbose_log("Removed JSON markdown suffix")
            response = response.strip()
            
            # Validate JSON
            try:
                self.verbose_log("Parsing JSON response...")
                json_response = json.loads(response)
                self.verbose_log("✅ JSON parsed successfully", "SUCCESS")
                
                # Detailed analysis of the response
                if self.verbose:
                    self.analyze_ai_response(json_response)
                
                # Print thinking process if enabled and not silent
                if self.show_thinking and not self.silent_mode:
                    self.verbose_log("📺 PHASE 5: DISPLAYING THINKING PROCESS", "INFO")
                    self.print_thinking_process(json_response)
                
                # Send comprehensive analysis to Discord
                if self.send_to_discord:
                    self.verbose_log("📨 PHASE 6: COMPREHENSIVE DISCORD NOTIFICATION", "INFO")
                    self.send_comprehensive_discord_analysis(json_response, client_data)
                
                final_message = json_response.get('message', 'No message generated')
                self.verbose_log("📱 FINAL RESULT", "SUCCESS")
                self.verbose_log(f"SMS Message: {final_message}")
                
                if not self.silent_mode:
                    print("\n📱 FINAL SMS MESSAGE:")
                    print("-" * 40)
                    print(f"📨 {final_message}")
                
                return json.dumps(json_response, indent=2, ensure_ascii=False)
                
            except json.JSONDecodeError as e:
                self.verbose_log(f"JSON parsing failed: {str(e)}", "ERROR")
                self.verbose_log("Attempting to extract message manually...")
                
                # If JSON parsing fails, extract message and create proper JSON
                message_match = re.search(r'"message":\s*"([^"]*)"', response)
                if message_match:
                    message = message_match.group(1)
                    self.verbose_log(f"Extracted message: {message}", "SUCCESS")
                    
                    # Send simple message to Discord if enabled
                    if self.send_to_discord:
                        simple_discord_msg = f"🏠 **Property Match Result**\n👤 Client ID: {client_id}\n📱 **Message:** {message}\n\n{self.get_verbose_logs_summary()}"
                        sendDiscordSanityCheckNoteAlert(simple_discord_msg, self.discord_channel_id)
                    
                    return json.dumps({"message": message} , ensure_ascii=False)
                else:
                    error_msg = "Unable to find suitable property match at this time."
                    self.verbose_log(f"No message found, using default: {error_msg}", "WARNING")
                    
                    if self.send_to_discord:
                        error_discord_msg = f"❌ **Property Match Error**\n👤 Client ID: {client_id}\n🚨 {error_msg}\n\n{self.get_verbose_logs_summary()}"
                        sendDiscordSanityCheckNoteAlert(error_discord_msg, self.discord_channel_id)
                    
                    return json.dumps({"message": error_msg} )
                    
        except Exception as e:
            self.verbose_log(f"Critical error in find_property_match: {str(e)}", "ERROR")
            
            if self.send_to_discord:
                error_discord_msg = f"🚨 **AGENT ERROR**\n👤 Client ID: {client_id}\n❌ Error: {str(e)}\n\n{self.get_verbose_logs_summary()}"
                sendDiscordSanityCheckNoteAlert(error_discord_msg, self.discord_channel_id)
            
            return json.dumps({"error": str(e)})

    def analyze_ai_response(self, json_response: Dict):
        """Analyze and log details about the AI response."""
        self.verbose_log("🔍 ANALYZING AI RESPONSE CONTENT:", "DEBUG")
        
        thinking = json_response.get("thinking_process", {})
        self.verbose_log(f"Thinking process sections: {len(thinking)}")
        
        for section, content in thinking.items():
            content_length = len(str(content))
            self.verbose_log(f"  - {section}: {content_length} characters")
            if content_length < 50:
                self.verbose_log(f"    WARNING: Short content in {section}", "WARNING")
        
        matches = json_response.get("property_matches", [])
        self.verbose_log(f"Property matches found: {len(matches)}")
        
        for i, match in enumerate(matches, 1):
            self.verbose_log(f"  Match {i}: {match.get('address', 'No address')} - Rank #{match.get('ranking', 'N/A')}")
        
        selected = json_response.get("selected_property", {})
        if selected:
            self.verbose_log(f"Selected property: {selected.get('address', 'No address')}")
            self.verbose_log(f"Price: {selected.get('price', 'No price')}")
            self.verbose_log(f"Neighborhood: {selected.get('neighborhood', 'No neighborhood')}")

    def calculate_neighborhood_proximity(self, client_address: str, property_address: str) -> float:
        """Calculate proximity score between client and property addresses."""
        try:
            client_location = self.geocoder.geocode(client_address)
            property_location = self.geocoder.geocode(property_address)
            
            if client_location and property_location:
                client_coords = (client_location.latitude, client_location.longitude)
                property_coords = (property_location.latitude, property_location.longitude)
                distance = geodesic(client_coords, property_coords).miles
                return distance
            return float('inf')
        except Exception:
            return float('inf')

# Main function for testing and execution
async def main(client_id: int = 689576, enable_discord: bool = True, verbose: bool = True, silent_mode: bool = False):
    """Main function to run the inventory agent."""
    try:
        if not silent_mode:
            print("🏠 NEIGHBORHOOD-FOCUSED PROPERTY MATCHING AGENT")
            print("=" * 80)
        
        agent = InventoryAgent(
            show_thinking=True, 
            send_to_discord=enable_discord,
            discord_channel_id='1379432907130802306',
            verbose=verbose,
            silent_mode=silent_mode
        )
        result = await agent.find_property_match(client_id)
        response_dict = json.loads(result)
        client_sms = response_dict.get("message", "")
        print("HELLLOOOOOODWUODWHIAUBDA AHHHHHHHHHHHHHHHHHHHH", client_sms)
        if not silent_mode:
            print("\n📋 COMPLETE ANALYSIS RESULT:")
            print("-" * 40)
            print(result)
        
        return result
    except Exception as e:
        error_msg = f"❌ Error in main: {str(e)}"
        if not silent_mode:
            print(error_msg)
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Example usage with comprehensive Discord logging
    client_id = 716237  # Replace with actual client ID
    result = asyncio.run(main(
        client_id=client_id,  
        enable_discord=True, 
        verbose=True,               # Capture all thinking logs
        silent_mode=False           # Set to True to suppress stdout printing
    ))
