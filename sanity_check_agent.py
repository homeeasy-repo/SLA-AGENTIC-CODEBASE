from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
import os
import json
from typing import Dict, Any, List
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()

# Get environment variables
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-2.0-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

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

def get_neighborhood_zip_codes_from_ai(neighborhood: str) -> List[str]:
    """Get ZIP codes for a neighborhood using Gemini AI."""
    try:
        # Create a prompt for the AI to get ZIP codes
        prompt = f"""
        You are a Chicago neighborhood expert. 
        What are the ZIP codes for the {neighborhood} neighborhood in Chicago, USA?
        Provide ONLY the ZIP codes in a comma-separated list format.
        Example format: 60601, 60602, 60603
        Do not include any other text or explanation.
        """
        
        # Get response from agent
        response = sanity_check_agent.run(prompt)
        
        # Extract ZIP codes from response
        zip_codes = []
        try:
            # Try to parse as comma-separated list
            zip_codes = [code.strip() for code in response.content.split(',')]
            # Filter out any non-ZIP code entries
            zip_codes = [code for code in zip_codes if code.isdigit() and len(code) == 5]
        except:
            # If parsing fails, try to extract numbers that look like ZIP codes
            import re
            zip_codes = re.findall(r'\b\d{5}\b', response.content)
        
        if not zip_codes:
            # Try one more time with a more specific prompt
            prompt = f"""
            You are a Chicago neighborhood expert. 
            For the {neighborhood} neighborhood in Chicago, USA, list ONLY the 5-digit ZIP codes.
            Format: 60601, 60602, 60603
            No other text, just ZIP codes.
            """
            response = sanity_check_agent.run(prompt)
            zip_codes = re.findall(r'\b\d{5}\b', response.content)
        
        return zip_codes
    except Exception as e:
        print(f"Error getting ZIP codes from AI: {str(e)}")
        return []

def get_neighborhood_zip_codes(neighborhood: str) -> List[Dict[str, Any]]:
    """Fetch building information using AI-provided ZIP codes."""
    session = None
    try:
        # Get ZIP codes from AI
        ai_zip_codes = get_neighborhood_zip_codes_from_ai(neighborhood)
        
        if not ai_zip_codes:
            raise ValueError(f"Could not determine ZIP codes for neighborhood: {neighborhood}")
        
        session = get_db_connection()
        # Use the AI-provided ZIP codes in the query
        query = text("""
            SELECT 
                zip_code,
                property_name,
                property_address,
                neighborhood_description,
                cooperating_status,
                cooperating_percentage,
                min_rent,
                max_rent,
                min_beds,
                max_beds
            FROM building_v2 
            WHERE zip_code = ANY(:zip_codes)
            AND zip_code IS NOT NULL
        """)
        results = session.execute(query, {"zip_codes": ai_zip_codes}).fetchall()
        
        # Format results
        neighborhood_info = []
        for result in results:
            info = {
                "zip_code": result[0],
                "property_name": result[1],
                "property_address": result[2],
                "neighborhood_description": result[3],
                "cooperating_status": result[4],
                "cooperating_percentage": result[5],
                "rent_range": f"${result[6] or 'N/A'} - ${result[7] or 'N/A'}",
                "bed_range": f"{result[8] or 'N/A'} - {result[9] or 'N/A'}"
            }
            neighborhood_info.append(info)
        
        if not neighborhood_info:
            raise ValueError(f"No buildings found in ZIP codes {ai_zip_codes} for neighborhood: {neighborhood}")
        
        return neighborhood_info
    
    except Exception as e:
        print(f"Error fetching neighborhood information: {str(e)}")
        raise
    finally:
        try:
            if session:
                session.close()
        except Exception as close_err:
            print(f"Warning: failed to close DB session: {close_err}")

def validate_neighborhood(neighborhood: str) -> Dict[str, Any]:
    """Validate neighborhood using AI-provided ZIP codes."""
    try:
        # Get ZIP codes from AI
        ai_zip_codes = get_neighborhood_zip_codes_from_ai(neighborhood)
        
        if not ai_zip_codes:
            return {
                "neighborhood": neighborhood,
                "zip_codes": [],
                "total_buildings": 0,
                "cooperating_buildings": {
                    "count": 0,
                    "percentage": 0
                },
                "buildings": [],
                "status": "⚠️",
                "message": f"Could not determine ZIP codes for {neighborhood}"
            }
        
        # Get building information using AI ZIP codes
        neighborhood_info = get_neighborhood_zip_codes(neighborhood)
        
        # Count cooperating buildings
        cooperating_buildings = [info for info in neighborhood_info if info['cooperating_status']]
        cooperating_count = len(cooperating_buildings)
        
        # Calculate average cooperating percentage
        avg_cooperating_percentage = sum(
            info['cooperating_percentage'] or 0 
            for info in cooperating_buildings
        ) / len(cooperating_buildings) if cooperating_buildings else 0
        
        return {
            "neighborhood": neighborhood,
            "zip_codes": ai_zip_codes,
            "total_buildings": len(neighborhood_info),
            "cooperating_buildings": {
                "count": cooperating_count,
                "percentage": round(avg_cooperating_percentage, 2) if cooperating_buildings else 0
            },
            "buildings": neighborhood_info,
            "status": "✅" if neighborhood_info else "⚠️",
            "message": f"Found {len(neighborhood_info)} buildings in ZIP codes {ai_zip_codes}" if neighborhood_info else f"No buildings found in ZIP codes {ai_zip_codes} for {neighborhood}"
        }
    
    except Exception as e:
        return {
            "neighborhood": neighborhood,
            "zip_codes": [],
            "total_buildings": 0,
            "cooperating_buildings": {
                "count": 0,
                "percentage": 0
            },
            "buildings": [],
            "status": "⚠️",
            "message": f"Error validating neighborhood: {str(e)}"
        }

# Create the Sanity Check Agent
sanity_check_agent = Agent(
    name="Sanity Check Agent",
    role="Real Estate Requirements Validator and Chicago Neighborhood Expert",
    model=Gemini(id=GENAI_MODEL, api_key=GOOGLE_API_KEY),
    add_name_to_instructions=True,
    instructions="""
    You are a real estate market expert and Chicago neighborhood specialist.
    Your task is to analyze client requirements and validate them against market standards.

    For each requirement field, evaluate:
    1. Is the value valid and realistic?
    2. Is any required field missing or undefined?
    3. Could the value cause rental eligibility issues?

    CHECK THRESHOLDS:
    - Credit Score: < 620 is concerning
    - Budget: Use client's provided budget range without assumptions if no upper limit provided just assume $100 more than the lower limit. If the client has a max budget, use that as the upper limit.
    - Move-in Date: Should be within 30 days for optimal availability
    - Bed/Bath: Should match typical unit configurations
    - Neighborhood: Should be valid areas with available inventory

    
    You must provide accurate ZIP codes for Chicago neighborhoods.
   
    BUDGET ANALYSIS BY NEIGHBORHOOD:
    For each requested neighborhood, analyze:
    1. Is the client's provided budget range realistic for that area?
    2. What is the average rent for the requested bed configuration in that neighborhood?
    3. Are there any properties in the inventory matching the client's budget range?
    4. Do not make assumptions about budget ranges - use only what the client has provided

    BED CONFIGURATION VALIDATION:
    For each requested bed configuration:
    1. Is this a common configuration in the requested neighborhoods?
    2. What is the typical price range for this bed configuration?
    3. Are there any properties in the inventory with this configuration?
    4. If configuration is rare, suggest common alternatives

    RESPONSE FORMAT:
    Return a JSON object with this exact structure:
    {
        "requirements_check": [
            {
                "field": "beds",
                "value": "current value",
                "status": "✅ or ⚠️",
                "comment": "brief explanation"
            },
            {
                "field": "baths",
                "value": "current value",
                "status": "✅ or ⚠️",
                "comment": "brief explanation"
            },
            {
                "field": "budget",
                "value": "current value",
                "status": "✅ or ⚠️",
                "comment": "brief explanation"
            },
            {
                "field": "move_in_date",
                "value": "current value",
                "status": "✅ or ⚠️",
                "comment": "brief explanation"
            },
            {
                "field": "neighborhood",
                "value": "current value",
                "status": "✅ or ⚠️",
                "comment": "brief explanation"
            },
            {
                "field": "credit_score",
                "value": "current value",
                "status": "✅ or ⚠️",
                "comment": "brief explanation"
            }
        ],
        "neighborhood_analysis": [
            {
                "neighborhood": "neighborhood name",
                "zip_codes": [zipcode],
                "budget_compatibility": "✅ or ⚠️",
                "average_rent": "typical rent range",
                "bed_configuration_availability": "availability status",
                "market_insights": "specific insights about this neighborhood"
            }
        ],
        "bed_configuration_analysis": [
            {
                "bed_count": "number of beds",
                "commonality": "how common this configuration is",
                "typical_price_range": "price range for this configuration",
                "availability": "availability status",
                "recommendations": "suggestions if needed"
            }
        ],
        "summary": {
            "concerns": ["List of major concerns that need follow-up"],
            "suggestions": ["List of suggestions for the client"],
            "questions": ["List of questions to ask the client"]
        }
    }

    IMPORTANT: Always include the zip_codes array for each neighborhood in your response.
    Keep the analysis factual and data-driven based on the inventory provided.
    Do not make assumptions about budget ranges - use only what the client has provided.
    """
)

async def check_requirements(client_id: int) -> Dict[str, Any]:
    """Run the sanity check on client requirements."""
    try:
        # Get requirements and inventory from database
        requirements = get_client_requirements(client_id)
        inventory = get_client_markdown_content(client_id)
        
        # Extract neighborhoods from requirements
        neighborhoods = []
        if requirements:
            for line in requirements.split('\n'):
                if 'Preferred Neighborhoods:' in line:
                    neighborhoods = [n.strip() for n in line.split(':')[1].split(',')]
        
        # Get ZIP codes for each neighborhood
        neighborhood_zip_mapping = {}
        for neighborhood in neighborhoods:
            zip_codes = get_neighborhood_zip_codes_from_ai(neighborhood)
            neighborhood_zip_mapping[neighborhood] = zip_codes
        
        # Validate neighborhoods
        neighborhood_validation = {}
        for neighborhood in neighborhoods:
            neighborhood_validation[neighborhood] = validate_neighborhood(neighborhood)
        
        # Create the prompt
        prompt = f"""
        CLIENT REQUIREMENTS:
        {requirements}

        AVAILABLE INVENTORY:
        {inventory}

        NEIGHBORHOOD ZIP CODES:
        {json.dumps(neighborhood_zip_mapping, indent=2)}

        NEIGHBORHOOD VALIDATION:
        {json.dumps(neighborhood_validation, indent=2)}

        Please analyze these requirements and provide your validation results.
        IMPORTANT: Include the ZIP codes for each neighborhood in your neighborhood_analysis response.
        
        For each neighborhood, you must include:
        1. The neighborhood name
        2. The ZIP codes array (from the NEIGHBORHOOD ZIP CODES section above)
        3. Budget compatibility analysis
        4. Average rent information
        5. Bed configuration availability
        6. Market insights
        """
        
        # Get response from agent
        response = sanity_check_agent.run(prompt)
        
        # Parse the response
        try:
            # First try direct JSON parsing
            result = json.loads(response.content)
            
            # Ensure ZIP codes are included in neighborhood analysis
            if 'neighborhood_analysis' in result:
                for analysis in result['neighborhood_analysis']:
                    neighborhood_name = analysis.get('neighborhood', '')
                    if neighborhood_name in neighborhood_zip_mapping:
                        analysis['zip_codes'] = neighborhood_zip_mapping[neighborhood_name]
                    elif 'zip_codes' not in analysis or not analysis['zip_codes']:
                        # Fallback: try to get ZIP codes again
                        zip_codes = get_neighborhood_zip_codes_from_ai(neighborhood_name)
                        analysis['zip_codes'] = zip_codes
            
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{[\s\S]*\}', response.content)
                if json_match:
                    result = json.loads(json_match.group(0))
                    
                    # Ensure ZIP codes are included
                    if 'neighborhood_analysis' in result:
                        for analysis in result['neighborhood_analysis']:
                            neighborhood_name = analysis.get('neighborhood', '')
                            if neighborhood_name in neighborhood_zip_mapping:
                                analysis['zip_codes'] = neighborhood_zip_mapping[neighborhood_name]
                else:
                    # If no JSON found, create a structured response
                    result = {
                        "requirements_check": [
                            {
                                "field": field,
                                "value": "Not provided",
                                "status": "⚠️",
                                "comment": "Unable to validate due to parsing error"
                            }
                            for field in ["beds", "baths", "budget", "move_in_date", "neighborhood", "credit_score"]
                        ],
                        "neighborhood_analysis": [
                            {
                                "neighborhood": neighborhood,
                                "zip_codes": neighborhood_zip_mapping.get(neighborhood, []),
                                "budget_compatibility": "⚠️",
                                "average_rent": "Unable to determine",
                                "bed_configuration_availability": "Unable to determine",
                                "market_insights": "Error parsing response"
                            }
                            for neighborhood in neighborhoods
                        ],
                        "summary": {
                            "concerns": ["Error parsing agent response"],
                            "suggestions": ["Please try again with different requirements"],
                            "questions": ["Would you like to try again?"]
                        }
                    }
            except Exception as e:
                print(f"Error parsing response: {str(e)}")
                print(f"Raw response: {response.content}")
                raise
            
        return result
            
    except Exception as e:
        print(f"Error in check_requirements: {str(e)}")
        return {
            "error": str(e),
            "requirements_check": [
                {
                    "field": field,
                    "value": "Not provided",
                    "status": "⚠️",
                    "comment": "Error during analysis"
                }
                for field in ["beds", "baths", "budget", "move_in_date", "neighborhood", "credit_score"]
            ],
            "summary": {
                "concerns": [f"Error during analysis: {str(e)}"],
                "suggestions": ["Please try again"],
                "questions": []
            }
        }

def print_results(result: Dict[str, Any]):
    """Print the sanity check results in a formatted way."""
    print("\nSanity Check Results:")
    print("=====================")
    
    # Print requirements check
    print("\nRequirements Validation:")
    for check in result.get('requirements_check', []):
        print(f"{check['status']} {check['field'].title()}: {check['value']} - {check['comment']}")
    
    # Print neighborhood analysis
    print("\nNeighborhood Analysis:")
    for analysis in result.get('neighborhood_analysis', []):
        print(f"\nNeighborhood: {analysis['neighborhood']}")
        print(f"ZIP Codes: {', '.join(analysis.get('zip_codes', []))}")
        print(f"Budget Compatibility: {analysis['budget_compatibility']}")
        print(f"Average Rent: {analysis['average_rent']}")
        print(f"Bed Configuration Availability: {analysis['bed_configuration_availability']}")
        print(f"Market Insights: {analysis['market_insights']}")
        
        # Print building listings
        if 'buildings' in analysis:
            print("\nAvailable Buildings:")
            for building in analysis['buildings']:
                print(f"\n📍 {building['property_name']}")
                print(f"   Address: {building['property_address']}")
                print(f"   ZIP: {building['zip_code']}")
                print(f"   Rent Range: {building['rent_range']}")
                print(f"   Bed Range: {building['bed_range']}")
                if building['cooperating_status']:
                    print(f"   Cooperating: Yes ({building['cooperating_percentage']}%)")
    
    # Print bed configuration analysis
    print("\nBed Configuration Analysis:")
    for analysis in result.get('bed_configuration_analysis', []):
        print(f"\nBed Count: {analysis['bed_count']}")
        print(f"Commonality: {analysis['commonality']}")
        print(f"Typical Price Range: {analysis['typical_price_range']}")
        print(f"Availability: {analysis['availability']}")
        print(f"Recommendations: {analysis['recommendations']}")
    
    # Print summary
    summary = result.get('summary', {})
    if summary.get('concerns'):
        print("\nConcerns:")
        for concern in summary['concerns']:
            print(f"⚠️ {concern}")
    
    if summary.get('suggestions'):
        print("\nSuggestions:")
        for suggestion in summary['suggestions']:
            print(f"💡 {suggestion}")
    
    if summary.get('questions'):
        print("\nQuestions to Ask:")
        for question in summary['questions']:
            print(f"❓ {question}")
    
    # Print concise summary
    print("\nQuick Summary:")
    print("=============")
    
    # Check overall status
    budget_compatible = all(analysis['budget_compatibility'] == '✅' 
                          for analysis in result.get('neighborhood_analysis', []))
    bed_available = all(analysis['availability'] == 'Available' 
                       for analysis in result.get('bed_configuration_analysis', []))
    all_requirements_valid = all(check['status'] == '✅' 
                               for check in result.get('requirements_check', []))
    
    # Print status line
    status = "✅" if all([budget_compatible, bed_available, all_requirements_valid]) else "⚠️"
    print(f"{status} Overall Status: {'Ready to proceed' if status == '✅' else 'Needs adjustment'}")
    
    # Print key findings
    if not budget_compatible:
        print(f"⚠️ Budget may need adjustment in some neighborhoods")
    if not bed_available:
        print(f"⚠️ Some bed configurations may be limited")
    if not all_requirements_valid:
        print(f"⚠️ Some requirements need review")
    
    # Print next step
    if status == '✅':
        print("Next: Proceed with property search")
    else:
        print("Next: Review and adjust requirements")

# Example usage
if __name__ == "__main__":
    async def main():
        try:
            # Get client ID from user
            client_id = int(input("Enter client ID: ").strip())
            
            if not client_id:
                print("Error: Client ID cannot be empty!")
                return
            
            # Run analysis
            print(f"\nAnalyzing requirements for client {client_id}...")
            result = await check_requirements(client_id)
            print(f"\nAnalysis Results for Client {client_id}:")
            print("=====================================")
            print_results(result)
            
        except ValueError as e:
            print(f"Error: {str(e)}")
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
    
    asyncio.run(main()) 