import random
import pandas as pd
from prompt import NodeManager
import os

# Example variations for user intents and assistant responses

# Define the system prompt
node_manager = NodeManager()
system_prompt = node_manager.get_navigation_map()

user_intents = [
    "Hi, I'm looking to schedule an appointment.",
    "Can I get a quote for fencing installation?",
    "What's your policy on cancellations?",
    "Do you offer any discounts for large projects?",
    "I want to change the phone number associated with my account.",
    "Can I transfer this call to someone else?",
    "What materials do you use for fences?",
    "Can you send someone tomorrow?",
    "I want to cancel the appointment I made last week.",
    "Can you explain your lifetime workmanship warranty?",
]

assistant_responses = [
    "Sure! Let me gather the details for you.",
    "Absolutely, I can assist with that.",
    "I can help you with that. Let me just pull up the information.",
    "Thanks for your question! I'll find that for you.",
    "I can connect you with the right person for that. Please hold on.",
    "Let me confirm your details before we proceed.",
    "I'll need a bit more information to assist you with that.",
    "Can you clarify your request so I can better assist you?",
    "Thank you for reaching out! Let's get that sorted for you.",
    "I'll take care of that right away. One moment, please.",
]

misleading_user_responses = [
    "Hmm, I'm not sure about that.",
    "Can I change my appointment to another day?",
    "What if I want to cancel later?",
    "Wait, I meant something else entirely.",
    "Can I talk to someone else about this?",
    "Oh, actually, I have another question.",
    "Does your policy cover this specific scenario?",
    "Can you clarify what you just said?",
    "I think I gave you the wrong details earlier.",
    "Can I add another service to my appointment?",
]

golden_responses = [
    "I understand your concern. Let's address your original request first.",
    "Sure, let me confirm your details to proceed with the appointment.",
    "Let me redirect you to the right step for this query.",
    "I'll ensure your request is handled. Let's continue with the appointment setup.",
    "I can provide clarity on that, but let's first confirm your original details.",
    "I'll answer your question and get you back on track for your request.",
    "Let me recheck your details to ensure we're moving forward correctly.",
    "I'll get that sorted for you. Let's finalize your current request first.",
    "Let me explain how our process works for this scenario.",
    "I'll guide you through this step to avoid any confusion.",
]

# Generate 50 diverse conversation scenarios
conversations = []
golden_responses_col = []

for i in range(50):
    conversation_history = []

    # Start the conversation
    conversation_history.append(
        {
            "role": "assistant",
            "content": "Hi! Thank you for calling Company X. How can I help you today?",
        }
    )
    user_query = random.choice(user_intents)
    conversation_history.append({"role": "user", "content": user_query})

    # Assistant's response based on user query
    assistant_reply = random.choice(assistant_responses)
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    # Add a misleading or ambiguous user response
    misleading_user_response = random.choice(misleading_user_responses)
    conversation_history.append({"role": "user", "content": misleading_user_response})

    # Perfect golden response to guide the conversation back
    golden_response = random.choice(golden_responses)
    conversation_history.append({"role": "assistant", "content": golden_response})

    # Append to the dataset
    conversations.append(conversation_history)
    golden_responses_col.append(golden_response)

# Create a DataFrame
data = {
    "System Prompt": [system_prompt] * 50,
    "Conversation History": conversations,
    "Golden Response": golden_responses_col,
}
df = pd.DataFrame(data)

# Save the dataset to a file
df.to_csv("./data/data.csv", index=False)

print("Dataset generated and saved to 'data.csv'")
