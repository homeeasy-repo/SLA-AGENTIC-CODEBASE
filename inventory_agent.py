import os
import datetime
from typing import Optional
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

# Database configuration
DATABASE_URL = "DATABASE_URL"

def get_db_connection():
    """Create and return a database connection."""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()

def format_client_requirements(requirements):
    """Format client requirements into a readable string."""
    req_list = []
    
    # Basic requirements
    if requirements.get('budget') or requirements.get('budget_max'):
        budget_range = f"${requirements.get('budget', '')}-${requirements.get('budget_max', '')}"
        req_list.append(f"- Budget: {budget_range} per month")
    
    if requirements.get('move_in_date'):
        req_list.append(f"- Move-in Date: {requirements['move_in_date']}")
    
    if requirements.get('beds') or requirements.get('baths'):
        unit_details = []
        if requirements.get('beds'):
            unit_details.append(f"{requirements['beds']} bedroom")
        if requirements.get('baths'):
            unit_details.append(f"{requirements['baths']} bathroom")
        req_list.append(f"- Unit Details: {' '.join(unit_details)}")
    
    # Location preferences
    if requirements.get('neighborhood'):
        neighborhoods = ', '.join(requirements['neighborhood'])
        req_list.append(f"- Location: {neighborhoods}")
    
    if requirements.get('zip'):
        zip_codes = ', '.join(requirements['zip'])
        req_list.append(f"- Zip Codes: {zip_codes}")
    
    # Additional preferences
    if requirements.get('parking'):
        req_list.append(f"- Parking: {requirements['parking']}")
    
    if requirements.get('pets'):
        req_list.append(f"- Pets: {requirements['pets']}")
    
    if requirements.get('washer_dryer'):
        req_list.append(f"- Washer/Dryer: {requirements['washer_dryer']}")
    
    if requirements.get('sqft') or requirements.get('sqft_max'):
        sqft_range = f"{requirements.get('sqft', '')}-{requirements.get('sqft_max', '')}"
        req_list.append(f"- Square Footage: {sqft_range}")
    
    if requirements.get('amenities'):
        amenities = ', '.join(requirements['amenities'])
        req_list.append(f"- Amenities: {amenities}")
    
    if requirements.get('building_must_haves'):
        req_list.append(f"- Building Must Haves: {requirements['building_must_haves']}")
    
    if requirements.get('unit_must_haves'):
        req_list.append(f"- Unit Must Haves: {requirements['unit_must_haves']}")
    
    if requirements.get('special_needs'):
        req_list.append(f"- Special Needs: {requirements['special_needs']}")
    
    if requirements.get('preference'):
        req_list.append(f"- Preferences: {requirements['preference']}")
    
    if requirements.get('comment'):
        req_list.append(f"- Additional Comments: {requirements['comment']}")
    
    return "\n".join(req_list)

def get_client_requirements(client_id: int) -> str:
    """Fetch client requirements from the database and format them."""
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
        session.close()

def get_api_key() -> str:
    """Get the Google API key from environment variable or raise an error."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please set it using: export GOOGLE_API_KEY='your-api-key'"
        )
    return api_key

# Initialize the Gemini model
try:
    genai.configure(api_key=get_api_key())
    model = GenerativeModel('gemini-1.5-pro')
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    raise

def analyze_properties(client_id: int, markdown_content: str) -> Optional[str]:
    """
    Analyze property listings and return the best match based on client requirements.
    
    Args:
        client_id (int): The ID of the client whose requirements to analyze.
        markdown_content (str): Markdown content containing property listings.
    
    Returns:
        Optional[str]: The analysis result in markdown format, or None if an error occurs.
    """
    try:
        # Get client requirements from database
        client_requirements = get_client_requirements(client_id)
        
        # Combine client requirements and markdown content into the prompt
        prompt = f"""
     You are a specialized real estate agent tasked with finding the perfect apartment match for a client based on their specific requirements and provided property listings in markdown format. Your goal is to analyze the listings thoroughly, select the best match, and independently craft a concise, engaging SMS draft to communicate with the client.

**Client Requirements:**
{client_requirements}

**Markdown Content (Multiple Property Listings):**
{markdown_content}

**Task:**
Analyze the provided property listings and select the best apartment that aligns with the client's requirements. Prioritize the following factors in order of importance:
1. **Budget**: Ensure the stated monthly rent falls within the client's specified budget.
2. **Move-in Date**: Confirm the property is available by the client's specified move-in date. Assume immediate availability if not specified.
3. **Location**: Match the property to the client's preferred neighborhood, city, and state. If specified, consider proximity to nearby amenities (e.g., public transit, grocery stores, parks).
4. **Unit Details**: Ensure the property meets the client's requirements for number of bedrooms, bathrooms, square footage, and any specific features (e.g., pet-friendly, in-unit laundry, parking).
5. **Move-in Specials**: Prioritize properties with attractive move-in specials (e.g., free weeks, discounts, reduced fees) that reduce upfront costs.
6. **Verified Listing**: Favor listings marked as verified for reliability.
7. **Locator Cooperation**: Prefer properties that cooperate with locators, if mentioned, to facilitate the leasing process.
8. **Additional Factors**: Consider unit size (square footage), deposit amounts, building amenities, and unique features that enhance value (e.g., balcony, gym, or low fees).

**Instructions:**
- If no property perfectly matches all requirements, select the closest match and explain any trade-offs or deviations.
- Note any missing or unclear information (e.g., pet policy, parking, availability, or amenities) in the response.
- Ensure all URLs (e.g., listing, property website, Google Maps) are included and functional, if provided.
- If proximity to amenities is specified, evaluate based on available information or note if data is insufficient.
- Verify that the listing is active and relevant based on provided data.
- Do not calculate or reference net effective rent (NER); use the stated monthly rent for budget comparisons.

**Output Format:**
Provide the details of the selected property in the following markdown format:
```
Name: [Property Name] | Address: [Full Address]
Management Company: [Management Company Name]
Management URL: [Management URL]
Rent Range: [Rent Range]
Phone Number: [Phone Number]
Listing URL: [Listing URL]
Property Website: [Property Website URL]
Move-in Special: [Move-in Special Details]
Verified Listing: [True/False]
Locator Cooperation: [Yes/No/Unknown]
Model: [Model Name] | Unit #: [Unit Number] | Model Price Range: [Price Range] | Bed/Bath: [Beds/Baths Details] | Sqft: [Square Footage] | Floor Plan: [Floor Plan Details] | Apply Here: [Application URL]
View on Google Maps: [Google Maps URL]
Nearby Amenities: [Details on proximity to client-specified amenities, if applicable, or note if data is missing]
Why This Property?: [Concise explanation of why this property is the best match, including alignment with client requirements, move-in specials, trade-offs, and unique features]
```

**SMS Draft for Client:**
As the agent, you must independently craft a concise, engaging SMS draft to introduce the selected property. Use a friendly, action-oriented tone, highlighting key benefits and prompting a response. The SMS must be under 160 characters :


**Notes:**
- Ensure the response is concise, comprehensive, and tailored to the client's needs.
- If multiple properties are close matches, select the one with the best overall alignment and briefly mention alternatives in the "Why This Property?" section.
- If critical information (e.g., move-in date, rent, or location) is missing, note it and make an informed assumption or recommend contacting the management company.
- Do not fabricate details not provided in the markdown content or client requirements.
- The SMS draft must be written by you, the agent, and remain engaging, professional, and within the 160-character limit to ensure compatibility with standard SMS protocols.
        ```
        """
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        print(f"Analysis error: {e}")
        return None

def get_client_markdown_content(client_id: int) -> str:
    """Fetch markdown content from notes for the client."""
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
        session.close()

def main():
    try:
        # Get client ID from input
        client_id = input("Please enter the client ID: ")
        try:
            client_id = int(client_id)
        except ValueError:
            print("Error: Client ID must be a number")
            return

        # Get markdown content from notes
        try:
            markdown_content = get_client_markdown_content(client_id)
            if not markdown_content or markdown_content.strip() == "":
                raise ValueError(f"Empty markdown content found for client_id: {client_id}")
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Analyze properties
        results = analyze_properties(client_id, markdown_content)

        if results:
            # Save summary to file
            filename = f"property_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w") as f:
                f.write("="*80 + "\n")
                f.write("PROPERTY ANALYSIS SUMMARY\n")
                f.write("="*80 + "\n\n")
                f.write(results)
                f.write("\n\n")

            print("PROPERTY ANALYSIS SUMMARY")
            print(f"\nFull summary saved to {filename}")
        else:
            print("No analysis results generated.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()