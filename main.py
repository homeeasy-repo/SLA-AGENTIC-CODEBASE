from agents import main_agent  # Assuming your MainAgent is inside agent.py

def main():
    print("🏡 Welcome to HomeEasy Sales Bot!")
    print("Type 'exit' anytime to quit.\n")

    while True:
        print("\n📩 Enter the conversation/chat history (client messages):")
        chat_history = input("> ")

        if chat_history.lower() in ["exit", "quit"]:
            break

        print("\n🏢 Enter the available inventory (property listings):")
        inventory_list = input("> ")

        if inventory_list.lower() in ["exit", "quit"]:
            break

        # Pack the input into a dict
        full_context = {
            "chat_history": chat_history,
            "inventory_list": inventory_list
        }

        # Call MainAgent
        sms_response = main_agent.process_query(full_context)

        print("\n📨 Final SMS to Client:")
        print(sms_response)
        print("\n" + "-"*70)

if __name__ == "__main__":
    main()
