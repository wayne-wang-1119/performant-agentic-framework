from openai import OpenAI
import pandas as pd
from prompt import NodeManager, PromptManager
import random
import os
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize NodeManager and PromptManager
node_manager = NodeManager()
prompt_manager = PromptManager(node_manager)

# Define the system prompt
system_prompt = prompt_manager.get(node_manager.get_navigation_map())

# Define user goals
user_goals = [
    "Schedule an appointment.",
    "Prank call.",
    "Ask about pricing for services.",
    "Inquire about cancellation policies.",
    "Update contact information.",
]


# Function to simulate conversation using LLM
def simulate_conversation(goal, system_prompt):
    conversation_history = []
    golden_response = ""

    # Initialize LLM with system prompt
    assistant_prompt = system_prompt
    user_prompt = f"You are a caller with the goal: {goal}. Start the conversation or based on the conversation history advance the conversation. Try to respond as human like as possible, which means you could likely change your idea, or have issues, or anything that is out of context."
    random_turns = random.randint(3, 9)
    model = "gpt-4o-mini"  # Specify your model
    for _ in range(random_turns):
        # User's input
        user_response = (
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": user_prompt},
                    *conversation_history,
                    {"role": "user", "content": "Your turn to respond."},
                ],
                stream=False,
            )
            .choices[0]
            .message.content
        )

        conversation_history.append({"role": "user", "content": user_response})

        # Agent's response
        assistant_response = (
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": assistant_prompt},
                    *conversation_history,
                    {"role": "assistant", "content": "Your turn to respond."},
                ],
                stream=False,
            )
            .choices[0]
            .message.content
        )

        conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        # Store golden response
        golden_response = assistant_response

        print(f"User: {user_response}")
        print(f"Assistant: {assistant_response}")

    return conversation_history, golden_response


# Generate dataset
conversations = []
golden_responses = []

for goal in user_goals:
    for _ in range(
        10
    ):  # Simulate 10 conversations per goal, totalling a 50 conversations
        convo_history, golden_response = simulate_conversation(goal, system_prompt)
        conversations.append(convo_history)
        golden_responses.append(golden_response)

# Create a DataFrame
data = {
    "System Prompt": [system_prompt] * len(conversations),
    "Conversation History": conversations,
    "Golden Response": golden_responses,
}
df = pd.DataFrame(data)

# Save the dataset to a file
output_path = "./data/simulated_data.csv"
df.to_csv(output_path, index=False)

print(f"Dataset generated and saved to '{output_path}'")
