from openai import OpenAI
import pandas as pd
from prompt_manager import NodeManager
from prompt_manager import PromptManager
import random
import os
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize NodeManager and PromptManager
node_manager = NodeManager()
prompt_manager = PromptManager(node_manager)
navigation_map = node_manager.get_submap_upto_node(0)

# Define the system prompt
system_prompt = prompt_manager.get()

# Define user goals
user_goals = [
    "Schedule an appointment.",
    "Prank call.",
    "Ask about services other than scheduling.",
    "Inquire about customer support that needs to be transferred.",
    "Update contact information.",
]


def call_llm_to_find_step(assistant_message, conversation_history, navigation_map):
    """
    Call the LLM to find the step or the Node ID that the agent is on currently
    based on the latest assistant message and conversation history.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are identifying which Node ID or step number most closely "
                "aligns with the latest assistant message. Return only the digit "
                "that represents the Step/Node ID."
            ),
        },
        {
            "role": "system",
            "content": """`
### Task Instructions:
You must identify the step from the navigation map that is **most similar** to the assistant's last response. The similarity should be based on the following criteria, in order of priority:
1. **Intent**: Match the primary purpose or action of the assistant's last response (e.g., confirming a name, providing an explanation, asking for information, etc.).
2. **Key Phrases**: Look for specific keywords or actions mentioned in the assistant's last response (e.g., "confirm," "name," "phone number").
3. **Context Alignment**: Consider how the assistant's last response aligns with the expected outcomes or instructions for each step in the navigation map.

### Steps to Determine the Most Similar Step:
1. **Understand the Assistant's Intent**: Analyze the assistant's last response to identify what action it is performing (e.g., confirming a name, asking for details, etc.).
2. **Analyze the Navigation Map**: Compare the intent and key phrases of the assistant's response with the instructions and expected behaviors for each step in the navigation map.
3. **Choose the Closest Match**: Select the step that most closely matches the intent and key phrases of the assistant's response. 

If multiple steps are similar, select the one with the closest **intent match**. Return the step number in JSON format.

### Additional Notes:
- For steps that contain instructions that end the call or indicate ending the call (e.g. "Ok, goodbye for now"), treat them with extra caution when selecting as a response. Since these steps end the call, they should typically appear only once. When you are evaluating potential next steps to return, avoid prematurely ending the call.
- If the conversation is not advancing to any appropriate step, return -1.
- When you return a step that is the end call step which has instruction that is end call message, you should only return that step if the latest assistant message is clearly the same as the end call message. 
- You should never return a step with instructions that do not resemble what the latest assistant message tries to achieve.

---

### Task:
Based on the navigation map, return the step that is most similar to what the AI assistant responded with in the last AI Message.
If the conversation is not advancing to any appropriate step, return -1. 
You should try to advance the conversation based on the latest assistant message.
""",
        },
        {
            "role": "system",
            "content": f"Here is the latest assistant message: {assistant_message}",
        },
        {
            "role": "system",
            "content": f"Here is the navigation map: {navigation_map}",
        },
    ]
    messages.extend(conversation_history)
    print("last assistant message:", assistant_message)
    print("conversation history:", conversation_history)
    print("navigation map:", navigation_map)

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    step_str = response.choices[0].message.content
    print("LLM response to find step:", step_str)
    return step_str


def format_user_flow_steps(flow_map):
    """
    Given a dictionary of steps (flow_map), return a string describing
    each step, its instruction, and its navigation options.
    """
    lines = []
    for step_number, step_info in flow_map.items():
        instruction = step_info.get("instruction", "No instruction found")
        line = f"On step {step_number} the agent will say back'{instruction}'."

        navigation = step_info.get("navigation")
        if isinstance(navigation, dict):
            for condition, next_step in navigation.items():
                line += f" The Agent will try to navigate you by '{condition}', you will be moved to {next_step}."
        elif isinstance(navigation, str):
            line += f" The Agent's action at this step will be: {navigation}."
        # else: no valid navigation - skip
        lines.append(line)
    return "\n".join(lines)


def format_ai_flow_steps(flow_map):
    """
    Given a dictionary of steps (flow_map), return a string describing
    each step, its instruction, and its navigation options.
    """
    lines = []
    for step_number, step_info in flow_map.items():
        instruction = step_info.get("instruction", "No instruction found")
        line = f"On step {step_number} you have instruction '{instruction}'."

        navigation = step_info.get("navigation")
        if isinstance(navigation, dict):
            for condition, next_step in navigation.items():
                line += f" Based on condition '{condition}', you can go to step {next_step}."
        elif isinstance(navigation, str):
            line += f" Navigation action: {navigation}."
        # else: no valid navigation - skip
        lines.append(line)
    return "\n".join(lines)


def format_convo_history(conversation_history):
    """
    Given a conversation history, format it as a string for display.
    """
    formatted_history = ""
    for turn in conversation_history:
        role = turn["role"]
        content = turn["content"]
        formatted_history += f"{"Caller" if role == "user" else "Agent"}: {content}\n"
    return formatted_history


# Function to simulate conversation using LLM
def simulate_conversation(goal, system_prompt, navigation_map):
    conversation_history = []
    golden_response = ""

    # Initialize LLM with system prompt
    assistant_prompt = system_prompt + "\n" + format_ai_flow_steps(navigation_map)
    user_sys_prompt = f"You are a caller with the goal: {goal}. Start the conversation or based on the conversation history advance the conversation. Try to respond as human like as possible, which means you could likely change your idea, or have issues, or anything that is out of context. You should start from now on generate a response that a caller would say instead of assistant message. If you are sending the first message to the agent then start with simple greeting that aligns to the goal implicitly."
    user_sys_prompt += f"The available options for you to continue the conversation based on the currrent options are:"
    user_prompt = user_sys_prompt + "\n" + format_user_flow_steps(navigation_map)
    random_turns = random.randint(6, 10)
    model = "gpt-4o"  # Specify your model
    print("=====================================================")
    for _ in range(random_turns):
        # User's input
        user_response = (
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": user_prompt},
                    {
                        "role": "system",
                        "content": "Current conversation history:"
                        + format_convo_history(conversation_history),
                    },
                    {
                        "role": "system",
                        "content": "Your turn to respond. Keep in mind that you are the caller, which is the user in the conversation history so far, instead of the assistant.",
                    },
                ],
                stream=False,
            )
            .choices[0]
            .message.content
        )

        conversation_history.append({"role": "user", "content": user_response})
        print(f"User: {user_response}")
        # Agent's response
        assistant_response = (
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": assistant_prompt},
                    *conversation_history,
                    {"role": "system", "content": "Your turn to respond."},
                ],
                stream=False,
            )
            .choices[0]
            .message.content
        )

        conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )
        print(f"Assistant: {assistant_response}")

        last_step_str = call_llm_to_find_step(
            assistant_response, conversation_history, navigation_map
        )

        if last_step_str != "-1":
            print(f"Last step: {last_step_str}")
            if type(last_step_str) == str:
                try:
                    last_step = int(re.findall(r"\d+", last_step_str)[0])
                except Exception:     
                    print("Error converting step to integer. Using 0.")
            else:
                last_step = last_step_str
            # Update the navigation map based on the last step
            navigation_map = node_manager.get_submap_upto_node(last_step)
            assistant_prompt = (
                system_prompt + "\n" + format_ai_flow_steps(navigation_map)
            )
            user_prompt = (
                user_sys_prompt + "\n" + format_user_flow_steps(navigation_map)
            )

            last_node_type = node_manager.full_map[last_step]
            if "terminate" in str(last_node_type):
                print("--------------------- Conversation ended. ---------------------")
                break

        # Store golden response
        golden_response = assistant_response

        print(f"Final User: {user_response}")
        print(f"Final Assistant: {assistant_response}")
    print("=====================================================")

    return conversation_history, navigation_map


# Function to determine the golden response based on the navigation map and the last agent message
def determine_golden_response(conversation_history, navigation_map):
    last_agent_message = conversation_history[-1]["content"]
    prompt = (
        f"Given the following navigation map and conversation history, "
        f"identify which node the agent is currently on based on its last message. "
        f"The navigation map outlines the agent's possible instructions and corresponding node IDs. "
        f"If the user requests a different service, the response should correspond to the relevant transfer node. "
        f"Always return only the original instruction from the identified node. "
        f"If no matching instruction exists, return an empty string.\n\n"
        f"Navigation Map:\n{navigation_map}\n\n"
        f"Conversation History:\n{format_convo_history(conversation_history)}\n\n"
        f"Return only the node instruction that the agent last delivered from the navigation map."
    )

    model = "gpt-4o"
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
        convo_history, updated_navigation_map = simulate_conversation(
            goal, system_prompt, navigation_map
        )
        golden_response = determine_golden_response(
            convo_history, updated_navigation_map
        )
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
