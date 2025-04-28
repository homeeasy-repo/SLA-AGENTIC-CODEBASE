# agent.py (start building this file)

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

GENAI_MODEL = os.getenv("GENAI_MODEL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class QualificationAgent:
    """Agent that qualifies the client by extracting motivation, urgency, and pain points using Socratic questioning."""

    def __init__(self):
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

    def setup_agent(self):
        """Initialize the Qualification Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="QualificationAnalyzer",
                func=self.analyze_qualification,
                description="Extracts client's motivation, urgency, and pain points by using Socratic questioning techniques."
            )
        ]

        system_message = """
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

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def analyze_qualification(self, query: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return query

    def process_query(self, client_message: str) -> str:
        """Process client input with system instructions directly using Gemini."""
        try:
            prompt = f"{self.system_message}\n\nClient Message: {client_message}\n\nBased on the client message above, analyze and return the qualification profile in the required output format."
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error in QualificationAgent: {str(e)}"

class ToneAgent:
    """Agent that decides the correct tone to use based on client qualification profile."""

    def __init__(self):
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

    def setup_agent(self):
        """Initialize the Tone Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="ToneSelector",
                func=self.select_tone,
                description="Decides whether to use a Concierge or Urgency tone based on client qualification."
            )
        ]

        system_message = """
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

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def select_tone(self, qualification_summary: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return qualification_summary

    def process_query(self, qualification_summary: str) -> str:
        """Pass the qualification profile to the Tone Agent and get a tone strategy."""
        try:
            response = self.agent.invoke({"input": qualification_summary})
            return response["output"]
        except Exception as e:
            return f"Error in ToneAgent: {str(e)}"


class InventoryAgent:
    """Agent that recommends properties based on client profile and urgency."""

    def __init__(self):
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

    def setup_agent(self):
        """Initialize the Inventory Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="InventoryMatcher",
                func=self.match_inventory,
                description="Matches client's profile and needs to available inventory, prioritizing high-commission, quick-close options."
            )
        ]

        system_message = """
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

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def match_inventory(self, client_profile: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return client_profile

    def process_query(self, client_profile: str) -> str:
        """Pass the client profile to the Inventory Agent and get matched property recommendations."""
        try:
            response = self.agent.invoke({"input": client_profile})
            return response["output"]
        except Exception as e:
            return f"Error in InventoryAgent: {str(e)}"


class ActionPlanAgent:
    """Agent that creates a structured action plan for both client and agent based on property matching and conversation."""

    def __init__(self):
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

    def setup_agent(self):
        """Initialize the Action Plan Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="ActionPlanBuilder",
                func=self.create_action_plan,
                description="Creates a time-bound action plan assigning tasks to the agent and the client based on matched inventory and client motivation."
            )
        ]

        system_message = """
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

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def create_action_plan(self, client_inventory_summary: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return client_inventory_summary

    def process_query(self, client_inventory_summary: str) -> str:
        """Pass the conversation and matched inventory summary to Action Plan Agent and get a structured action plan."""
        try:
            response = self.agent.invoke({"input": client_inventory_summary})
            return response["output"]
        except Exception as e:
            return f"Error in ActionPlanAgent: {str(e)}"


class ObjectionHandlerAgent:
    """Agent that handles client objections using HomeEasy-approved fact-based, urgency-driven, and psychology-grounded techniques."""

    def __init__(self):
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

    def setup_agent(self):
        """Initialize the Objection Handling Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="ObjectionResolver",
                func=self.handle_objection,
                description="Handles client objections like price concerns, hesitations, delays, by providing fact-based logical rebuttals."
            )
        ]

        system_message = """
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

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def handle_objection(self, objection_message: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return objection_message

    def process_query(self, objection_message: str) -> str:
        """Pass the objection message to the Objection Handler Agent and get a rebuttal + next step."""
        try:
            response = self.agent.invoke({"input": objection_message})
            return response["output"]
        except Exception as e:
            return f"Error in ObjectionHandlerAgent: {str(e)}"


class ApplicationCloserAgent:
    """Agent that drives the client to complete the application process, explains next steps, and creates urgency."""

    def __init__(self):
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

    def setup_agent(self):
        """Initialize the Application Closer Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="ApplicationCloser",
                func=self.close_application,
                description="Pushes the client to complete the rental application after property match is done. Explains next steps clearly and frames application positively."
            )
        ]

        system_message = """
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

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def close_application(self, application_prompt: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return application_prompt

    def process_query(self, application_prompt: str) -> str:
        """Pass the final client + property context to Application Closer Agent and get application instructions."""
        try:
            response = self.agent.invoke({"input": application_prompt})
            return response["output"]
        except Exception as e:
            return f"Error in ApplicationCloserAgent: {str(e)}"

class PostApplicationAgent:
    """Agent that manages post-application activities: payment confirmation, lease signing, move-in coordination, and ongoing client communication."""

    def __init__(self):
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

    def setup_agent(self):
        """Initialize the Post-Application Follow-Up Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="PostApplicationFollowUp",
                func=self.follow_up_application,
                description="Manages post-application follow-ups: payment verification, lease generation, move-in scheduling, and maintaining client communication until move-in is complete."
            )
        ]

        system_message = """
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

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def follow_up_application(self, post_application_context: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return post_application_context

    def process_query(self, post_application_context: str) -> str:
        """Pass the application context to Post-Application Agent and get next follow-up actions."""
        try:
            response = self.agent.invoke({"input": post_application_context})
            return response["output"]
        except Exception as e:
            return f"Error in PostApplicationAgent: {str(e)}"

class SMSFormatterAgent:
    """Agent that formats all outgoing messages into short, natural, human-like SMS replies, optimized for client communication."""

    def __init__(self):
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

    def setup_agent(self):
        """Initialize the SMS Formatter Agent with the correct tool and system instructions."""
        tools = [
            Tool(
                name="SMSFormatter",
                func=self.format_sms,
                description="Formats structured agent responses into short, natural, human-like SMS messages under 300 characters if possible."
            )
        ]

        system_message = """
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

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def format_sms(self, full_response: str) -> str:
        """Simple passthrough for now since LangChain needs a function."""
        return full_response

    def process_query(self, full_response: str) -> str:
        """Pass the full response to the SMS Formatter Agent and get a short SMS-ready text."""
        try:
            response = self.agent.invoke({"input": full_response})
            return response["output"]
        except Exception as e:
            return f"Error in SMSFormatterAgent: {str(e)}"

class MainAgent:
    """Main coordinating agent that orchestrates all specialized agents and generates the final SMS-ready response."""

    def __init__(self):
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
                func=lambda input: self.inventory_agent.process_query(
                    input.get("client_profile", ""), 
                    input.get("inventory_list", "")
                ),
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

        system_message = """
            You are HomeEasy's Main Sales Coordinator Assistant.

            You act as a **real human leasing consultant** — friendly, respectful, professional.

            ---
            GENERAL BEHAVIOR:

            - Always begin the conversation politely.
            - If the client expresses confusion (e.g., "Who are you?" / "Who gave you my info?"):
            - Respond politely: "We received your information through one of the properties we work with. We're here to help you find your next apartment."
            - Always respect client privacy — never reveal detailed private information.
            - Always sound confident, warm, and helpful.
            - Never behave like an AI — you are a real person helping with rentals.
            ---

            AGENT CALLING RULES:

            1. **Start with Introduction**:
            - If conversation is early or confused, simply craft an introduction and format it via SMSFormatterAgent.

            2. **Qualification**:
            - If you have no clear details about client's move-in motivation, call **QualificationAgent** to understand needs.

            3. **Tone Decision**:
            - After qualifying, call **ToneAgent** to set whether you will use "Concierge" or "Urgency" tone in future replies.

            4. **Inventory Matching**:
            - If inventory is available, call **InventoryAgent** with the client profile and available units.
            - Recommend 3–5 best units politely.

            5. **Action Plan**:
            - After matching units, call **ActionPlanAgent** to structure next steps with clear deadlines.

            6. **Objection Handling**:
            - If client raises concerns (e.g., "Price too high", "Need to think"), call **ObjectionHandlerAgent** to craft respectful rebuttals.

            7. **Application Push**:
            - If client selects a property, call **ApplicationCloserAgent** to push application respectfully.

            8. **Post-Application Followup**:
            - After application submission, call **PostApplicationAgent** to manage move-in coordination.

            ---
            SPECIAL INSTRUCTIONS:

            - At the END of **any conversation**, ALWAYS pass your response to **SMSFormatterAgent** to format it properly as a short SMS reply.

            - When passing to SMSFormatterAgent, ask it to:
            - Respectfully maintain politeness.
            - Summarize clearly under 300 characters if possible.
            - Ensure message sounds human and not scripted.

            ---
            FINAL GOAL:

            Your mission is to:
            - Engage the client politely
            - Understand their rental needs
            - Guide them step-by-step toward signing an apartment lease
            - Always sound human, smart, and respectful
            - Always return only one short final SMS-ready response.

            ---

            Remember:
            **Always guide, never just answer. Always format, never skip SMSFormatterAgent.**

            Act like a real HomeEasy consultant at every moment.
        """

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

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
            # Merge chat_history + inventory into a single string
            combined_input = f"""
Client Conversation History:
{full_context.get('chat_history', '')}

Available Inventory:
{full_context.get('inventory_list', '')}
"""

            # Pass the combined input to the agent
            structured_response = self.agent.invoke({"input": combined_input})
            structured_message = structured_response["output"]

            # Then SMSFormatterAgent trims it into clean SMS
            sms_final = self.sms_formatter_agent.process_query(structured_message)

            return sms_final.strip()
        except Exception as e:
            return f"Error in MainAgent: {str(e)}"
        

#### AGENT CALLING ####

qualification_agent = QualificationAgent()
tone_agent = ToneAgent()
inventory_agent = InventoryAgent()
action_plan_agent = ActionPlanAgent()
objection_handler_agent = ObjectionHandlerAgent()
application_closer_agent = ApplicationCloserAgent()
post_application_agent = PostApplicationAgent()
sms_formatter_agent = SMSFormatterAgent()
main_agent = MainAgent()