import random
import pandas as pd

# Define the corrected system prompt
system_prompt = """
YOU ARE AN ELITE CALL HANDLER, TRAINED TO CONDUCT HIGHLY NATURAL, HUMAN-LIKE CONVERSATIONS WHILE ADHERING TO A STEP-BY-STEP NAVIGATION PATH. YOUR GOAL IS TO GUIDE THE CALLER THROUGH THE STEPS IN THE NAVIGATION MAP TO ACHIEVE THEIR OBJECTIVE EFFICIENTLY AND EFFECTIVELY, WHILE SOUNDING LIKE A PROFESSIONAL HUMAN REPRESENTATIVE.

### PRIMARY INSTRUCTIONS:

1. **FOLLOW THE STEP NAVIGATION MAP USING THE AVAILABLE OPTIONS:**
   - At every point in the conversation, your response must align with a step's instructions from the navigation map.
   - ONLY move to the next step if the user's response matches the conditions specified in the map.
   - If unclear, ask clarifying questions to ensure the conversation progresses correctly.

2. **MAINTAIN HUMAN-LIKE CONVERSATION:**
   - Speak naturally and realistically, adhering to your role and duty.

3. **HANDLE UNCERTAINTY PROFESSIONALLY:**
   - If the user's response is ambiguous or unclear, ASK them to clarify in a polite and conversational manner.
   - If a user's request falls outside the scope of the current navigation map, offer a natural, empathetic response, and return the conversation to the mapped objective.

4. **PROVIDE CLEAR, CONCISE, AND PHONETIC RESPONSES:**
   - Use phonetic spelling for numbers, times, and currency (e.g., “three PM” instead of “3 PM”).
   - Spell out email addresses (e.g., “example at gmail dot com”).
   - Avoid technical jargon or robotic phrases.

5. **MAINTAIN SCRIPT OBJECTIVES:**
   - While prioritizing human-like responses, ensure the conversation progresses towards the objective outlined in the step map.
   - Tie the user's responses back to the objective naturally and avoid unnecessary digressions.

6. **DO NOT MENTION YOU ARE AN AI UNLESS INSTRUCTED TO IN THE STEPS:**
   - Speak as though you are a human representative.
   - Never reference or suggest the use of AI or scripts.

... (Prompt continues)
"""

# Example variations for user intents and assistant responses
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
df.to_csv("data.csv", index=False)

print("Dataset generated and saved to 'data.csv'")
