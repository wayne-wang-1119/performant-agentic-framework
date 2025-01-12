import os
import re
import ast
import json
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

from openai import OpenAI
from prompt_manager import NodeManager
from prompt_manager import PromptManager

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------- Step 1: Remove SentenceTransformer usage -------------------
# model = SentenceTransformer("all-MiniLM-L6-v2")  # <-- Removed


# ------------------- Step 2: Add a helper function to call OpenAI Embeddings API -------------------
def vectorize_prompt(model: str, prompt_text: str) -> List[float]:
    """
    Call OpenAI's Embeddings API with the given `model` and `prompt_text`.
    Returns a list of floats corresponding to the embedding.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    request_body = {"input": prompt_text, "model": model}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    url = "https://api.openai.com/v1/embeddings"
    response = requests.post(url, json=request_body, headers=headers)

    if response.status_code != 200:
        raise ValueError(
            f"OpenAI API returned status {response.status_code}: {response.text}"
        )

    data = response.json()
    if "data" not in data or len(data["data"]) == 0:
        raise ValueError("No data in OpenAI API response")

    # Return the first embedding
    embedding = data["data"][0]["embedding"]
    return embedding


# ------------------- NodeManager & Navigation Map -------------------
# Instantiate NodeManager. Make sure your NodeManager has also been updated
# to store embeddings via OpenAI, so that node_manager.node_embeddings is populated.
node_manager = NodeManager()
navigation_map = node_manager.get_navigation_map()


def call_llm(system_prompt, conversation_history, user_message):
    """
    Calls the LLM with the system prompt, conversation history, and user message.
    Returns the generated assistant response.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


def clean_response(response):
    """
    Clean up quotes and whitespace from a string.
    """
    if isinstance(response, str):
        response = response.strip('"')
        response = response.strip()
    return response


# ------------------- Step 3: Update compute_semantic_similarity to use vectorize_prompt -------------------
def compute_semantic_similarity(response_1, response_2):
    """
    Compute semantic similarity between two responses using cosine similarity
    via OpenAI embeddings.
    """
    # Get embeddings from OpenAI
    emb1 = vectorize_prompt("text-embedding-3-small", response_1)
    emb2 = vectorize_prompt("text-embedding-3-small", response_2)

    # Use sklearn's cosine_similarity
    similarity = cosine_similarity([emb1], [emb2])[0][0]

    # Clamp the similarity to [0.0, 1.0]
    similarity = max(0.0, min(similarity, 1.0))
    return similarity


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

### Examples:
#### Example 1:
- **Assistant's Last Message**: "So just to confirm, your name is John Doe, and the phone number is 123-456-7890. Is that correct?"
- **Most Similar Step**: Step 46 (Confirming name or details)
- **Reason**: The intent of confirming a name aligns directly with the instructions in Step 46.

#### Example 2:
- **Assistant's Last Message**: "Okay, let's cancel your existing appointment. Can you provide the phone number or email you used to book the appointment?"
- **Most Similar Step**: Step 5 (Requesting phone number or email to cancel an appointment)
- **Reason**: The action of requesting phone/email to cancel matches Step 5.

#### Example 3:
- **Assistant's Last Message**: "Got it, unfortunately, we cannot book teen appointments over the phone."
- **Most Similar Step**: Step 9 (Informing the user about teen appointment booking policy)
- **Reason**: The message aligns with the specific instructions for handling teen bookings in Step 9.

#### Example 4:
- ** Conversation History **:
- **Assistant's Last Message**: "You are all set! Your appointment is confirmed for three PM on December fifth. Is there anything else I can assist you with before we wrap  up? "
- **User's Last Message**: "yeah like i mean can you double check my number i just wanna make sure you got it right "
- **Navigation Map**:
On step 5, you should say: "[Confirm the caller has no more questions]". 
On step 9, you should say: "Ok goodbye for now."
- **Most Similar Step**: Step 5
- **Reason**: The assistant's last message aligns with the expected behavior for Step 5, which involves confirming the caller has no more questions before wrapping up the call.

### Additional Notes:
- For steps that contain instructions that end the call or indicate ending the call (e.g. “Ok, goodbye for now”), treat them with extra caution when selecting as a response. Since these steps end the call, they should typically appear only once. When you are evaluating potential next steps to return, avoid prematurely ending the call.
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


# ------------------- Step 4: Update find_step_with_vectors to use OpenAI embeddings instead of model.encode(...) -------------------
def find_step_with_vectors(assistant_message: str) -> Tuple[int, float]:
    """
    Vectorize assistant_message and compare it to each node's embedding
    using the dot product or cosine similarity.
    Returns (best_node_id, best_score).
    """
    # Get the embedding for the message
    embedding = vectorize_prompt("text-embedding-3-small", assistant_message)

    best_node_id = None
    best_score = float("-inf")

    # Compare to each node embedding in node_manager
    for node_id, node_emb in node_manager.node_embeddings.items():
        # Use cosine similarity
        score = cosine_similarity([embedding], [node_emb])[0][0]
        # Clamp the similarity to [0.0, 1.0]
        score = max(0.0, min(score, 1.0))
        if score > best_score:
            best_score = score
            best_node_id = node_id

    # If best_score > 0.96, we consider that "good enough"; otherwise we return None
    if best_score > 0.96:
        return best_node_id, best_score
    return None, 0.0


def format_flow_steps(flow_map):
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


# ------------------- Main code to load CSV, process, evaluate similarity -------------------
INPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/eval_paf.csv")

df = pd.read_csv(INPUT_FILE)

# Standardize column names if needed
df.rename(
    columns={
        "System Prompt": "system_prompt",
        "Conversation History": "convo_history",
        "Golden Response": "golden_response",
    },
    inplace=True,
)

semantic_similarities = []

for idx, row in df.iterrows():
    try:
        system_prompt = row["system_prompt"]
        convo_history_str = row["convo_history"]
        golden_response_str = row["golden_response"]

        convo_history = ast.literal_eval(convo_history_str)
        golden_response = clean_response(ast.literal_eval(golden_response_str))

        if not convo_history:
            print(f"Row {idx}: Empty conversation history. Skipping.")
            semantic_similarities.append(None)
            continue

        messages = [
            convo_history[0],
            convo_history[1],
        ]  # Add the first user message
        generated_response = None

        # We'll keep a "current_system_prompt" that can get updated with submap info
        current_system_prompt = system_prompt
        current_navi_map = format_flow_steps(node_manager.get_navigation_map())

        i = 1
        while i < len(convo_history):
            turn = messages[i]

            # If the turn is from the assistant (dataset's assistant),
            # we add it to the conversation context, then run the step-finder.
            if turn["role"] == "assistant":
                assistant_msg = turn["content"]

                # 1) Step finder logic: vector method + LLM method
                vector_step_id, vector_step_score = find_step_with_vectors(
                    assistant_msg
                )
                llm_step_str = call_llm_to_find_step(
                    assistant_msg, messages, current_navi_map
                )

                # 2) Decide which step to use (vector vs LLM)
                if vector_step_id is not None:
                    step_str = str(vector_step_id)
                    print(
                        "Using vector method for step:",
                        vector_step_id,
                        vector_step_score,
                    )
                else:
                    step_str = llm_step_str
                    print("Using LLM method for step:", llm_step_str)

                # 3) Convert step_str to int
                try:
                    step_identifier = int(re.findall(r"\d+", step_str)[0])
                except Exception:
                    step_identifier = 0

                # 4) Build submap and update system prompt
                submap = node_manager.get_submap_upto_node(step_identifier)
                current_navi_map = format_flow_steps(submap)
                current_system_prompt = (
                    f"{system_prompt}\n\n"
                    f"You were at step {step_identifier} based on the latest assistant message.\n"
                    f"Below is a partial navigation map relevant to your current step:\n{current_navi_map}\n\n"
                    "Now continue from that context."
                )
                if i + 1 < len(convo_history):
                    messages.append(convo_history[i + 1])

            elif turn["role"] == "user":
                # 1) Append user message
                user_msg = turn["content"]
                messages.append({"role": "user", "content": user_msg})

                # 2) Call LLM to get new assistant message
                assistant_reply = call_llm(current_system_prompt, messages, user_msg)
                assistant_reply = clean_response(assistant_reply)

                generated_response = assistant_reply
                messages.append({"role": "assistant", "content": assistant_reply})

                # 3) Step finder logic
                vector_step_id, vector_step_score = find_step_with_vectors(
                    assistant_reply
                )
                llm_step_str = call_llm_to_find_step(
                    assistant_reply, messages, current_navi_map
                )

                if vector_step_id is not None:
                    step_str = str(vector_step_id)
                    print(
                        "Using vector method for step:",
                        vector_step_id,
                        vector_step_score,
                    )
                else:
                    step_str = llm_step_str
                    print("Using LLM method for step:", llm_step_str)

                try:
                    step_identifier = int(re.findall(r"\d+", step_str)[0])
                except Exception:
                    step_identifier = 0

                submap = node_manager.get_submap_upto_node(step_identifier)
                current_navi_map = format_flow_steps(submap)
                current_system_prompt = (
                    f"{system_prompt}\n\n"
                    f"You were at step {step_identifier} based on the latest assistant message.\n"
                    f"Below is a partial navigation map relevant to your current step:\n{current_navi_map}\n\n"
                    "Now continue from that context."
                )

                print("=====================================================")
                print("Map we are using:", current_navi_map)
                print("System prompt we are using:", current_system_prompt)

            i += 1

        # 8) Evaluate similarity
        similarity_score = compute_semantic_similarity(
            generated_response, golden_response
        )

        print("=====================================================")
        print(f"Processed row {idx + 1}: Similarity = {similarity_score}")
        print(f"Generated response: {generated_response}")
        print(f"Golden response: {golden_response}")
        print("=====================================================")

        semantic_similarities.append(similarity_score)

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        semantic_similarities.append(None)

# 9) Write results
df["optimized_semantic_similarity"] = semantic_similarities
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")
