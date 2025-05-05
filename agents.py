import logging
import os
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GENAI_MODEL = os.getenv("GENAI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Utility function for message formatting
def format_gemini_messages(system_message: str, user_input: str) -> list:
    """Format input as a list of messages for Gemini API."""
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]

class QualificationAgent:
    """Agent that qualifies the client by extracting motivation, urgency, and pain points using Socratic questioning."""

    def __init__(self):
        self.system_message = """
            You are a HomeEasy Leasing Consultant specializing in client qualification.

            Your mission is to deeply understand why a client wants to move by asking strategic Socratic questions. 
            You must extract the client's motivation, timeline, and pain points without assuming anything. 

            Responsibilities:
            1. Start by asking probing open-ended questions like:
            - "What’s making you consider moving at this time?"
            - "Why not stay where you are?"
            - "What is your ideal move-in date?"
            2. Identify emotional or logical triggers (lease ending, job relocation, dissatisfaction).
            3. Build a complete profile of the client's urgency and motivations.
            4. NEVER make assumptions. Always extract the information by asking.
            5. Maintain a neutral, professional tone — not pushy or overly casual.
            6. DO NOT offer inventory yet — focus only on understanding client psychology.

            Final Output Format:
            - Motivation Summary: (1-2 lines)
            - Urgency Level: (High / Medium / Low)
            - Timeline: (Move-in deadline, if provided)
            - Pain Points Identified: (List)

            Goal:
            Prepare this profile cleanly for the next sales phase.

            Always remember: "Extract statements; don’t make them."
        """
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.4,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agent()
        logger.info("QualificationAgent initialized")

    def setup_agent(self):
        """Initialize the Qualification Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="QualificationAnalyzer",
                func=self.analyze_qualification,
                description="Extracts client's motivation, urgency, and pain points by using Socratic questioning techniques."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def analyze_qualification(self, query: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return query

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process_query(self, client_message: str) -> str:
        """Process client input with system instructions directly using Gemini."""
        try:
            if not client_message.strip():
                logger.error("Client message is empty")
                return "Error: Client message cannot be empty."

            prompt = f"Client Message: {client_message}\n\nBased on the client message above, analyze and return the qualification profile in the required output format."
            messages = format_gemini_messages(self.system_message, prompt)
            logger.info("QualificationAgent processing: %s", client_message)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("Error in QualificationAgent: %s", str(e))
            return f"Error in QualificationAgent: {str(e)}"

class ToneAgent:
    """Agent that decides the correct tone to use based on client qualification profile."""

    def __init__(self):
        self.system_message = """
            You are a HomeEasy Tone Calibration Advisor.

            Your mission is to choose the correct communication style for each client based on their qualification profile.

            Responsibilities:
            1. If the client is highly qualified (high credit score, strong income):
            - Use a soft, consultative, concierge tone.
            - Focus on premium service, options, and personalization.

            2. If the client is lower qualified (lower credit, lower income):
            - Use a high-pressure, urgency-driven tone.
            - Emphasize scarcity, deadlines, and decision-making urgency.

            3. If the qualification is unclear, ask strategic questions to clarify.

            Final Output Format:
            - Tone Style: (Concierge / Urgency)
            - Reason for Choice: (short 1-2 sentence justification)
            - Example Opening Sentence: (based on chosen tone)

            Goal:
            Prepare the proper psychological strategy to maximize conversion.

            Always remember: "Speak in the client's interest, but guide them firmly."
        """
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.4,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agent()
        logger.info("ToneAgent initialized")

    def setup_agent(self):
        """Initialize the Tone Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="ToneSelector",
                func=self.select_tone,
                description="Decides whether to use a Concierge or Urgency tone based on client qualification."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def select_tone(self, qualification_summary: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return qualification_summary

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process_query(self, qualification_summary: str) -> str:
        """Pass the qualification profile to the Tone Agent and get a tone strategy."""
        try:
            if not qualification_summary.strip():
                logger.error("Qualification summary is empty")
                return "Error: Qualification summary cannot be empty."

            messages = format_gemini_messages(self.system_message, qualification_summary)
            logger.info("ToneAgent processing: %s", qualification_summary)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("Error in ToneAgent: %s", str(e))
            return f"Error in ToneAgent: {str(e)}"

class InventoryAgent:
    """Agent that recommends properties based on client profile and urgency."""

    def __init__(self):
        self.system_message = """
            You are a HomeEasy Inventory Matching Specialist.

            Your mission is to suggest the best rental options based on the client's motivation, urgency, budget, and preferences.

            Responsibilities:
            1. Match the client to 3–5 properties based on:
               - Budget
               - Desired location
               - Urgency (ready-to-move vs flexible)
               - Property features needed (e.g., washer/dryer, 3-bedrooms)
            2. Prioritize:
               - High-commission properties
               - Units that can close fastest (e.g., same day, guest card application units)
               - Available and vacant units over future openings.
            3. If no perfect match, suggest creative options:
               - Studios instead of 1-bed if needed
               - Nearby neighborhoods
               - Different lease lengths

            Final Output Format:
            - List 3-5 Property Suggestions:
              - Name
              - Key features
              - Rent amount
              - Move-in availability
              - Why it's a good fit (1-2 lines)

            Goal:
            Recommend fast-close inventory first to maximize conversion and revenue.

            Always remember: "Help the client make the fastest, safest, smartest choice."
        """
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.4,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agent()
        logger.info("InventoryAgent initialized")

    def setup_agent(self):
        """Initialize the Inventory Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="InventoryMatcher",
                func=self.match_inventory,
                description="Matches client's profile and needs to available inventory, prioritizing high-commission, quick-close options."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def match_inventory(self, client_profile: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return client_profile

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process_query(self, client_profile: str) -> str:
        """Pass the client profile to the Inventory Agent and get matched property recommendations."""
        try:
            if not client_profile.strip():
                logger.error("Client profile is empty")
                return "Error: Client profile cannot be empty."

            messages = format_gemini_messages(self.system_message, client_profile)
            logger.info("InventoryAgent processing: %s", client_profile)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("Error in InventoryAgent: %s", str(e))
            return f"Error in InventoryAgent: {str(e)}"

class ActionPlanAgent:
    """Agent that creates a structured action plan for both client and agent based on property matching and conversation."""

    def __init__(self):
        self.system_message = """
            You are a HomeEasy Action Plan Creator.

            Your mission is to generate a clear, specific, time-bound action plan after matching properties to the client.

            Responsibilities:
            1. Define specific next steps for BOTH sides:
               - Agent Tasks (e.g., send property links, confirm showings, book appointments)
               - Client Tasks (e.g., review properties, send documents, confirm showings)
            2. Always assign **DEADLINES** (example: "Today by 5:00 PM").
            3. Ensure that each action step is **urgent** but **achievable**.
            4. Make sure tasks are logical based on:
               - Client urgency
               - Matched property options
               - Rental application process

            Final Output Format:
            - **Agent Tasks:**
              - [Task 1] (Deadline)
              - [Task 2] (Deadline)

            - **Client Tasks:**
              - [Task 1] (Deadline)
              - [Task 2] (Deadline)

            - **Next Meeting / Follow-up Scheduled:** (Specific time, example: "Tomorrow at 10 AM via call or SMS")

            Goal:
            Create momentum and urgency by structuring the entire conversation into next steps.

            Always remember: "Time is the enemy — act quickly."
        """
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.4,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agent()
        logger.info("ActionPlanAgent initialized")

    def setup_agent(self):
        """Initialize the Action Plan Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="ActionPlanBuilder",
                func=self.create_action_plan,
                description="Creates a time-bound action plan assigning tasks to the agent and the client based on matched inventory and client motivation."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def create_action_plan(self, client_inventory_summary: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return client_inventory_summary

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process_query(self, client_inventory_summary: str) -> str:
        """Pass the conversation and matched inventory summary to Action Plan Agent and get a structured action plan."""
        try:
            if not client_inventory_summary.strip():
                logger.error("Client inventory summary is empty")
                return "Error: Client inventory summary cannot be empty."

            messages = format_gemini_messages(self.system_message, client_inventory_summary)
            logger.info("ActionPlanAgent processing: %s", client_inventory_summary)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("Error in ActionPlanAgent: %s", str(e))
            return f"Error in ActionPlanAgent: {str(e)}"

class ObjectionHandlerAgent:
    """Agent that handles client objections using HomeEasy-approved fact-based, urgency-driven, and psychology-grounded techniques."""

    def __init__(self):
        self.system_message = """
            You are a HomeEasy Objection Handling Specialist.

            Your mission is to **overcome client objections** using logical reasoning, fact-based corrections, urgency creation, and emotional reassurance.

            Responsibilities:
            1. Identify the root cause of the objection (budget concern, hesitation, fear of making wrong decision, market misunderstanding).
            2. Respond using structured techniques:
               - Reframe the objection constructively.
               - Justify price or urgency using real-world market facts.
               - Emphasize scarcity ("units moving fast") when necessary.
               - Offer alternatives creatively if needed (e.g., different unit types, different locations).
               - Remain empathetic but **never passive** — **guide to decision**.

            Common Objection Examples:
            - "It's too expensive" → (Show value justification: savings on amenities, location, etc.)
            - "I need to think" → (Create urgency: "units getting leased daily")
            - "I found cheaper options online" → (Explain differences, quality, approval ease)

            Final Output Format:
            - Rebuttal Response: (2-4 sentences)
            - Suggested Next Step: (e.g., "Let's submit an application now to hold your spot.")

            Tone:
            - Friendly but firm
            - Logical
            - Structured
            - Client-focused (always in their best interest)

            Goal:
            Convert hesitation into a clear next step toward closing.

            Always remember: "Frame facts as opportunities, not criticisms."
        """
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.4,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agent()
        logger.info("ObjectionHandlerAgent initialized")

    def setup_agent(self):
        """Initialize the Objection Handling Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="ObjectionResolver",
                func=self.handle_objection,
                description="Handles client objections like price concerns, hesitations, delays, by providing fact-based logical rebuttals."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def handle_objection(self, objection_message: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return objection_message

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process_query(self, objection_message: str) -> str:
        """Pass the objection message to the Objection Handler Agent and get a rebuttal + next step."""
        try:
            if not objection_message.strip():
                logger.error("Objection message is empty")
                return "Error: Objection message cannot be empty."

            messages = format_gemini_messages(self.system_message, objection_message)
            logger.info("ObjectionHandlerAgent processing: %s", objection_message)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("Error in ObjectionHandlerAgent: %s", str(e))
            return f"Error in ObjectionHandlerAgent: {str(e)}"

class ApplicationCloserAgent:
    """Agent that drives the client to complete the application process, explains next steps, and creates urgency."""

    def __init__(self):
        self.system_message = """
            You are a HomeEasy Application Closing Specialist.

            Your mission is to smoothly and professionally **move the client into the application phase** after matching properties.

            Responsibilities:
            1. Congratulate the client for reaching the application stage.
            2. Outline the application process clearly:
               - What documents are needed (ID, paystubs, etc.)
               - Application fees (if any)
               - Next steps after applying (approval, lease signing)
            3. Create **urgency**:
               - Remind that inventory moves fast.
               - First-come, first-served — completing application locks their spot.
            4. Offer to help if needed (e.g., guide them through online application portal).

            Final Output Format:
            - Application Push Message: (2–4 sentences)
            - Document Checklist: (short list if needed)
            - Reminder of Urgency: (short sentence)
            - Closing Line: (friendly encouragement to act now)

            Tone:
            - Warm
            - Logical
            - Supportive
            - Professional

            Goal:
            Make the client feel excited, confident, and ready to submit their application.

            Always remember: "Frame the next step as a victory, not a burden."
        """
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.4,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agent()
        logger.info("ApplicationCloserAgent initialized")

    def setup_agent(self):
        """Initialize the Application Closer Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="ApplicationCloser",
                func=self.close_application,
                description="Pushes the client to complete the rental application after property match is done. Explains next steps clearly and frames application positively."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def close_application(self, application_prompt: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return application_prompt

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process_query(self, application_prompt: str) -> str:
        """Pass the final client + property context to Application Closer Agent and get application instructions."""
        try:
            if not application_prompt.strip():
                logger.error("Application prompt is empty")
                return "Error: Application prompt cannot be empty."

            messages = format_gemini_messages(self.system_message, application_prompt)
            logger.info("ApplicationCloserAgent processing: %s", application_prompt)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("Error in ApplicationCloserAgent: %s", str(e))
            return f"Error in ApplicationCloserAgent: {str(e)}"

class PostApplicationAgent:
    """Agent that manages post-application activities: payment confirmation, lease signing, move-in coordination, and ongoing client communication."""

    def __init__(self):
        self.system_message = """
            You are a HomeEasy Post-Application Follow-Up Specialist.

            Your mission is to **escort the client from application submission all the way to successful move-in**.

            Responsibilities:
            1. Confirm the client submitted payment (application fee, security deposit).
            2. Follow up daily with the building until:
               - Lease is generated and sent
               - Lease is signed
               - Keys are collected
            3. Update the client proactively every 24–48 hours (even if no major news).
            4. Escalate issues if building is unresponsive (e.g., contact sister properties or managers).
            5. Maintain positivity and urgency to keep the process moving forward.

            Final Output Format:
            - Payment Verification Status
            - Lease Status Update
            - Move-In Scheduling Status
            - Next Client Communication Plan

            Tone:
            - Proactive
            - Reassuring
            - Structured
            - High-urgency when needed

            Goal:
            Ensure the client successfully moves in, feels supported throughout the process, and experiences zero confusion.

            Always remember: "The sale is not complete until the keys are in the client's hand."
        """
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.4,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agent()
        logger.info("PostApplicationAgent initialized")

    def setup_agent(self):
        """Initialize the Post-Application Follow-Up Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="PostApplicationFollowUp",
                func=self.follow_up_application,
                description="Manages post-application follow-ups: payment verification, lease generation, move-in scheduling, and maintaining client communication until move-in is complete."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def follow_up_application(self, post_application_context: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return post_application_context

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process_query(self, post_application_context: str) -> str:
        """Pass the application context to Post-Application Agent and get next follow-up actions."""
        try:
            if not post_application_context.strip():
                logger.error("Post-application context is empty")
                return "Error: Post-application context cannot be empty."

            messages = format_gemini_messages(self.system_message, post_application_context)
            logger.info("PostApplicationAgent processing: %s", post_application_context)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("Error in PostApplicationAgent: %s", str(e))
            return f"Error in PostApplicationAgent: {str(e)}"

class SMSFormatterAgent:
    """Agent that formats all outgoing messages into short, natural, human-like SMS replies, optimized for client communication."""

    def __init__(self):
        self.system_message = """
            You are a HomeEasy SMS Formatting Specialist.

            Your mission is to **convert structured agent responses** into **short, clear, human-sounding SMS drafts**.

            Responsibilities:
            1. Trim long texts without losing important meaning.
            2. Keep the tone friendly, professional, and efficient.
            3. Ensure SMS sounds human — avoid robotic or overly formal language.
            4. Keep messages under 300 characters if possible.
            5. If trimming is necessary, prioritize clarity and next steps.

            Examples:
            - Instead of: "Please be advised that we would like to schedule a tour."
            - Say: "Hey! When would be a good time for your tour? 😊"

            Final Output Format:
            - SMS Draft: (1-2 lines, ready to be sent via phone)

            Tone:
            - Warm
            - Human
            - Professional
            - Slightly casual if appropriate

            Goal:
            Make the client feel they are texting a real human leasing agent.

            Always remember: "SMS = Short, Meaningful, Swift."
        """
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.4,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agent()
        logger.info("SMSFormatterAgent initialized")

    def setup_agent(self):
        """Initialize the SMS Formatter Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="SMSFormatter",
                func=self.format_sms,
                description="Formats structured agent responses into short, natural, human-like SMS messages under 300 characters if possible."
            )
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )

    def format_sms(self, full_response: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return full_response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process_query(self, full_response: str) -> str:
        """Pass the full response to the SMS Formatter Agent and get a short SMS-ready text."""
        try:
            if not full_response.strip():
                logger.error("Full response is empty")
                return "Error: Full response cannot be empty."

            messages = format_gemini_messages(self.system_message, full_response)
            logger.info("SMSFormatterAgent processing: %s", full_response)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("Error in SMSFormatterAgent: %s", str(e))
            return f"Error in SMSFormatterAgent: {str(e)}"

class MainAgent:
    """Main coordinating agent that orchestrates all specialized agents and generates the final SMS-ready response."""

    def __init__(self):
        self.system_message = """
            You are HomeEasy's Main Sales Coordinator, a professional, friendly leasing consultant who communicates like a human.

            ---

            **Core Behavior**:
            - Begin every interaction with a polite, warm tone.
            - If the client is confused (e.g., "Who are you?"), respond: "Hey! I'm with HomeEasy, here to help you find your next apartment!"
            - Protect client privacy; never share sensitive details.
            - Avoid AI-like responses; act as a trusted, real person.

            ---

            **Agent Calling Rules**:

            1. **Introduction**:
               - For new or vague messages (e.g., "Who are you?" or "Looking for an apartment"), craft a friendly greeting and pass to **SMSFormatterAgent**.

            2. **Qualification**:
               - If client needs (budget, location, size, move-in date) are unclear, call **QualificationAgent** to ask targeted questions (e.g., "What's your budget and ideal location?").

            3. **Tone Selection**:
               - After qualification, call **ToneAgent** to set "Concierge" (high-qualified) or "Urgency" (lower-qualified) tone.

            4. **Inventory Matching**:
               - If client needs and inventory are available, call **InventoryAgent** to suggest 3–5 properties matching budget, location, and preferences.

            5. **Action Plan**:
               - Post-inventory, call **ActionPlanAgent** to outline urgent, actionable next steps for agent and client.

            6. **Objection Handling**:
               - For hesitations (e.g., "Too expensive"), call **ObjectionHandlerAgent** to provide empathetic, logical rebuttals.

            7. **Application Push**:
               - If a property is chosen, call **ApplicationCloserAgent** to encourage application submission.

            8. **Post-Application Follow-Up**:
               - After application, call **PostApplicationAgent** to manage lease signing and move-in.

            ---

            **Special Instructions**:
            - Always route the final response through **SMSFormatterAgent** for a polite, human-like SMS (aim for <300 characters).
            - For vague or incomplete inputs, ask clarifying questions (e.g., "What size apartment and budget are you looking for?").
            - If no inventory matches, inform the client and ask for more details or flexibility.

            ---

            **Goal**:
            Engage clients warmly, clarify their rental needs, and guide them toward a lease with concise, SMS-formatted responses.

            ---

            **Key Reminder**:
            Proactively guide the conversation, always use **SMSFormatterAgent**, and embody a trusted HomeEasy consultant.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=GENAI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.4
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.setup_agents()
        self.setup_coordinator()
        logger.info("MainAgent initialized")

    def setup_agents(self):
        """Initialize all specialized agents."""
        self.qualification_agent = QualificationAgent()
        self.tone_agent = ToneAgent()
        self.inventory_agent = InventoryAgent()
        self.action_plan_agent = ActionPlanAgent()
        self.objection_handler_agent = ObjectionHandlerAgent()
        self.application_closer_agent = ApplicationCloserAgent()
        self.post_application_agent = PostApplicationAgent()
        self.sms_formatter_agent = SMSFormatterAgent()

    def setup_coordinator(self):
        """Setup the main agent to coordinate everything."""
        tools = [
            Tool(
                name="QualificationAgent",
                func=self.qualification_agent.process_query,
                description="Extracts client motivation, urgency, and pain points."
            ),
            Tool(
                name="ToneAgent",
                func=self.tone_agent.process_query,
                description="Decides correct communication tone (Concierge or Urgency)."
            ),
            Tool(
                name="InventoryAgent",
                func=self.inventory_agent.process_query,
                description="Matches properties to client profile and recommends inventory."
            ),
            Tool(
                name="ActionPlanAgent",
                func=self.action_plan_agent.process_query,
                description="Creates structured action plans assigning tasks to agent and client."
            ),
            Tool(
                name="ObjectionHandlerAgent",
                func=self.objection_handler_agent.process_query,
                description="Handles client objections logically and factually."
            ),
            Tool(
                name="ApplicationCloserAgent",
                func=self.application_closer_agent.process_query,
                description="Pushes the client to complete the rental application process."
            ),
            Tool(
                name="PostApplicationAgent",
                func=self.post_application_agent.process_query,
                description="Manages post-application follow-ups: lease signing, move-in."
            ),
            Tool(
                name="SMSFormatterAgent",
                func=self.sms_formatter_agent.process_query,
                description="Formats the final response into short, human-like SMS."
            ),
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )
        logger.info("MainAgent coordinator initialized with tools: %s", [tool.name for tool in tools])

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process_query(self, full_context: dict) -> str:
        """
        Process incoming conversation + inventory with proper routing.
        `full_context` should be a dict:
        {
            "chat_history": "....",
            "inventory_list": "...."
        }
        """
        try:
            if not full_context.get("chat_history"):
                logger.error("Chat history is missing")
                return "Error: Chat history is required."

            combined_input = f"""
Client Conversation History:
{full_context.get('chat_history', '')}

Available Inventory:
{full_context.get('inventory_list', '')}
"""
            logger.info("MainAgent processing: %s", combined_input)
            # messages = format_gemini_messages(self.system_message, combined_input)
            # structured_response = self.agent.invoke(messages)
            # structured_message = structured_response["output"]
            agent_prompt = f"{self.system_message}\n\n{combined_input}"
            structured_message = self.agent.run(agent_prompt)

            # sms_final = self.sms_formatter_agent.process_query(structured_message)
            # return sms_final.strip()
            sms_final = self.sms_formatter_agent.process_query(structured_message)

            # ✅ Remove hallucinated 'Example' blocks if present
            if "Example 1" in sms_final or "Structured Agent Response" in sms_final:
                logger.warning("Detected hallucinated training examples. Replacing with clarification message.")
                return "Hey! Could you please clarify what you're looking for in an apartment so I can help better? 😊"

            return sms_final.strip()
        except Exception as e:
            logger.error("Error in MainAgent: %s", str(e))
            return f"Error in MainAgent: {str(e)}"

# Agent instantiation
qualification_agent = QualificationAgent()
tone_agent = ToneAgent()
inventory_agent = InventoryAgent()
action_plan_agent = ActionPlanAgent()
objection_handler_agent = ObjectionHandlerAgent()
application_closer_agent = ApplicationCloserAgent()
post_application_agent = PostApplicationAgent()
sms_formatter_agent = SMSFormatterAgent()
main_agent = MainAgent()


# Test the MainAgent
test_context = {
    "chat_history": "Client: Hi, I'm looking for a new apartment. What's your name?",
    "inventory_list": "Inventory: 1-bedroom, 1 bath, $1500-$2000, 1 mile from downtown"
}

response = main_agent.process_query(test_context)
print(response.content)
