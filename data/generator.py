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
    "Ask about services other than scheduling.",
    "Inquire about customer support that needs to be transferred.",
    "Update contact information.",
]


# Function to simulate conversation using LLM
def simulate_conversation(goal, system_prompt):
    conversation_history = []
    golden_response = ""

    # Initialize LLM with system prompt
    assistant_prompt = system_prompt
    user_prompt = f"You are a caller with the goal: {goal}. Start the conversation or based on the conversation history advance the conversation. Try to respond as human like as possible, which means you could likely change your idea, or have issues, or anything that is out of context. You should start from now on generate a response that a caller would say instead of assistant message. If you are sending the first message to the agent then start with simple greeting that aligns to the goal implicitly."
    random_turns = random.randint(2, 5)
    model = "gpt-4o"  # Specify your model
    print("=====================================================")
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
    print("=====================================================")

    return conversation_history, golden_response


# Function to determine the golden response based on the navigation map and the last agent message
def determine_golden_response(conversation_history, system_prompt):
    last_agent_message = conversation_history[-1]["content"]

    prompt = f"Based on the following navigation map and system prompt, identify the node instruction that most closely aligns with the last agent message. If the user requests a different service, ensure the golden response corresponds to the transfer node. Always and only return the original instruction of the node that is being said by the Agent, if it does not exist then return empty string. Do not return anything else. System Prompt:\n{system_prompt}\n\nLast Agent Message:\n{last_agent_message}\n\nGolden Response:"  # Explicit prompt for LLM

    model = "gpt-4o"  # Specify your model
    golden_response = (
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            stream=False,
        )
        .choices[0]
        .message.content
    )

    return golden_response


# Generate dataset
conversations = []
golden_responses = []

for goal in user_goals:
    for _ in range(10):  # Simulate 10 conversations per goal, totaling 50 conversations
        convo_history, _ = simulate_conversation(goal, system_prompt)
        golden_response = determine_golden_response(convo_history, system_prompt)
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
output_path = "./data/dataset.csv"
df.to_csv(output_path, index=False)

print(f"Dataset generated and saved to '{output_path}'")


# Manually go through the golden response and update each to the correct Node responses.
