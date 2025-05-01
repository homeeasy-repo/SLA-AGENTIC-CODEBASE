import os
import datetime
from phi.agent import Agent
from phi.model.google import Gemini
from typing import Optional

# Define the real estate agent
agent = Agent(
    model=Gemini(id="gemini-1.5-pro", api_key=os.environ.get("GOOGLE_API_KEY")),
    description="You are a specialized real estate agent tasked with finding the perfect apartment match for a client based on their specific requirements and provided property listings in markdown format.",
    instructions = [
    "Analyze the provided markdown content containing multiple apartment listings.",
    "Select the **best property** that aligns with the client's requirements, prioritizing:",
    "- **Budget**: Ensure the rent falls within the client's specified budget, using net effective rent (NER) if move-in specials are offered.",
    "- **Move-in Date**: Confirm the property is available by the client's move-in date (assume immediate availability if not specified).",
    "- **Location**: Match the property to the client's preferred neighborhood, city, and state.",
    "- **Unit Details**: Ensure the property meets the client's requirements for bedrooms and bathrooms.",
    "- **Move-in Specials**: Prioritize properties with attractive move-in specials (e.g., free weeks, discounts) that reduce upfront costs.",
    "- **Net Effective Rent Calculation**: If a move-in special is mentioned (e.g., 'X weeks free'), calculate the net effective rent (NER) over a 12-month lease using the formula: `NER = (Monthly Rent * 12 - (Monthly Rent * Free Months)) / 12`. Use this NER value to determine if the property falls within budget and provides better value.",
    "- **Additional Factors**: Consider unit size (square footage), deposit amounts, and any unique features or amenities that enhance value.",
    "If no property perfectly matches, select the closest match and explain why it was chosen.",
    "Note any missing or unclear information (e.g., pet policy, parking, or availability) in the response.",
    "Provide the details of the selected property in the following markdown format:",
    "```",
    "Name: [Property Name] | Address: [Full Address]",
    "Management Company: [Management Company Name]",
    "Management URL: [Management URL]",
    "Rent Range: [Rent Range]",
    "Phone Number: [Phone Number]",
    "Listing URL: [Listing URL]",
    "Property Website: [Property Website URL]",
    "Move-in Special: [Move-in Special Details]",
    "Verified Listing: [True/False]",
    "Model: [Model Name] | Unit #: [Unit Number] | Model Price Range: [Price Range] | Bed/Bath: [Beds/Baths Details] | Sqft: [Square Footage] | Floor Plan: [Floor Plan Details] | Apply Here: [Application URL]",
    "View on Google Maps: [Google Maps URL]",
    "Why This Property?: [Explanation of why this property is the best match, referencing client requirements, move-in specials, calculated NER, and unique features]",
    "```",
    "Ensure the response is concise yet comprehensive, tailored to the client's needs.",
    "The client requirements and markdown content are provided as input."
],


    markdown=True,
    show_tool_calls=False,  # No external tools used
    add_datetime_to_instructions=True,
)

def analyze_properties(client_requirements: str, markdown_content: str) -> Optional[str]:
    """
    Analyze property listings and return the best match based on client requirements.
    
    Args:
        client_requirements (str): The client's apartment requirements.
        markdown_content (str): Markdown content containing property listings.
    
    Returns:
        Optional[str]: The analysis result in markdown format, or None if an error occurs.
    """
    try:
        # Combine client requirements and markdown content into the prompt
        prompt = f"""
        **Client Requirements:**
        {client_requirements}

        **Markdown Content (Multiple Property Listings):**
        {markdown_content}
        """
        
        # Run the agent with the prompt
        response = agent.run(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"Analysis error: {e}")
        return None

def main():
    try:
        # Set required environment variable for Gemini API key
        os.environ["GOOGLE_API_KEY"] ="GOOGLE_API_KEY"
  # Replace with your API key

        # Sample client requirements
        client_requirements = """
        - Budget: $1000-$1300 per month
        - Move-in Date: Immediate
        - Location: Garland, TX or nearby (within 5 miles)
        - Unit Details: 1 bedroom, 1 bathroom
        - Preferences: Move-in specials preferred, pet-friendly, parking available
        """

        # Sample markdown content with property listings
        markdown_content = """
        Name: Filament | Address: 4689 Saturn Rd, Garland, TX 75041
        Management Company: Avenue5 Residential
        Management URL: https://www.apartments.com/pmc/avenue5-residential/3we23ce/
        Rent Range: $1399.0 - $2314.0
        Phone Number: (833) 891-2482
        Listing URL: https://www.apartments.com/filament-garland-tx/hbsrsp1/
        Property Website: https://www.liveatfilament.com/
        Move-in Special: NOW LEASING! 1 month FREE plus an additional $250 off move-in!
        Verified Listing: True
        Model: A1 | Unit #: 2219 | Model Price Range: $1,344 – $1,414 | Bed/Bath: 1 Bed 1 Bath 622 Sq Ft | Sqft: 622 | Floor Plan: NaN | Apply Here: http://liveatfilament.securecafe.com/
        View on Google Maps: http://www.google.com/maps/search/?api=1&query=Filament%2C%204689%20Saturn%20Rd%2C%20Garland%2C%20TX%2075041
        ------------------------------------
        Name: Atwater | Address: 5026 Zion Rd, Garland, TX 75043
        Management Company: Zrs Management
        Management URL: https://www.apartments.com/pmc/zrs-management/lyh9fhf/
        Rent Range: $1239.0 - $3445.0
        Phone Number: (469) 517-0025
        Listing URL: https://www.apartments.com/atwater-garland-tx/q5k0mqz/
        Property Website: https://www.atlascrowntx.com/
        Move-in Special: 8 weeks free! Lease with us TODAY!
        Model: A2.1 | Unit #: 4205 | Model Price Range: $1,385 – $1,560 | Bed/Bath: 1 Bed 1 Bath 713 Sq Ft $500 Deposit | Sqft: 713 | Floor Plan: NaN | Apply Here: NaN
        View on Google Maps: http://www.google.com/maps/search/?api=1&query=Atwater%2C%205026%20Zion%20Rd%2C%20Garland%2C%20TX%2075043
        ------------------------------------
        Name: The Quinn on Thirty | Address: 6302 Greenbelt, Garland, TX 75043
        Management Company: Garland Tx
        Management URL: https://www.apartments.com/pmc/garland-tx/
        Rent Range: $1299.0 - $2105.0
        Phone Number: (469) 850-7124
        Listing URL: https://www.apartments.com/the-quinn-on-thirty-garland-tx/b3426fc/
        Move-in Special: Receive up to 8 weeks free! Additional bonus may apply!
        Verified Listing: True
        Model: A1 | Unit #: 1419 | Model Price Range: $1,299 – $1,434 | Bed/Bath: 1 Bed 1 Bath 656 Sq Ft $100 Deposit | Sqft: 656
        View on Google Maps: http://www.google.com/maps/search/?api=1&query=The%20Quinn%20on%20Thirty%2C%206302%20Greenbelt%2C%20Garland%2C%20TX%2075043
        """

        # Analyze properties
        results = analyze_properties(client_requirements, markdown_content)

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