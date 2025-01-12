import os
import pandas as pd
import ast
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from prompt_manager import NodeManager
from prompt_manager import PromptManager

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load pre-trained model for semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

node_manager = NodeManager()
navigation_map = node_manager.get_navigation_map()

# File paths
INPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/eval_base.csv")


def call_llm(system_prompt, conversation_history, user_message):
    """
    Calls the LLM with the system prompt, conversation history, and user message.
    Returns the generated assistant response.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # Example call to a non-standard model "gpt-4o-mini"
    # Adjust to whichever model or endpoint you're actually using
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


def compute_semantic_similarity(response_1, response_2):
    """
    Compute semantic similarity between two responses using cosine similarity.
    """
    embeddings = model.encode([response_1, response_2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    # Clamp the similarity to [0.0, 1.0]
    similarity = max(0.0, min(similarity, 1.0))
    return similarity


def call_llm_to_find_step(assistant_message, conversation_history, navigation_map):
    """
    Call the LLM to find the step or the Node id that the agent is on currently based on the latest assistant message and conversation history.
    """
    messages = [
        {
            "role": "system",
            "content": "You are identifying which Node ID or step number most closely aligns with the latest assistant message. Return only the digit that represents where the Step or Node ID you think matches the condition described.",
        },
        {
            "role": "system",
            "content": "Here is the latest assistant message:" + assistant_message,
        },
        {
            "role": "system",
            "content": "Here is the navigation map:" + str(navigation_map),
        },
    ]
    messages.extend(conversation_history)
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    # Extract the step number from the generated response
    print("LLM response to find step:", response.choices[0].message.content)
    return response.choices[0].message.content


# Load and preprocess dataset
df = pd.read_csv(INPUT_FILE)

# Standardize column names
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

        # Convert stored string data back to Python objects
        convo_history = ast.literal_eval(convo_history_str)
        golden_response = clean_response(ast.literal_eval(golden_response_str))

        if not convo_history:
            print(f"Row {idx}: Empty conversation history. Skipping.")
            semantic_similarities.append(None)
            continue

        # We'll build the conversation *turn by turn*, calling the LLM whenever
        # we see a user message. We will also call the step-finder logic
        # *immediately after* each new assistant response.

        # 1) We start a local `messages` list that will accumulate everything
        messages = [convo_history[0]]  # Add the first assistant message
        generated_response = None

        # 2) Prepare an initial system prompt that can be updated each turn
        #    if we want to insert step info along the way.
        current_system_prompt = system_prompt

        i = 0
        while i < len(convo_history):
            turn = messages[i]

            if turn["role"] == "assistant":
                messages.append({"role": "assistant", "content": turn["content"]})
                step_id = call_llm_to_find_step(
                    turn["content"], messages, navigation_map
                )
                current_system_prompt = (
                    f"{system_prompt}\n\n"
                    f"You were at step {step_id} based on the latest assistant message.\n"
                    "Now continue from that context."
                )

            elif turn["role"] == "user":
                # 1) Append user message
                user_msg = turn["content"]
                messages.append({"role": "user", "content": user_msg})

                # 2) Call LLM to get new assistant response
                assistant_reply = call_llm(current_system_prompt, messages, user_msg)
                assistant_reply = clean_response(assistant_reply)

                generated_response = assistant_reply

                # 3) Append that assistant reply to our messages
                messages.append({"role": "assistant", "content": assistant_reply})

            i += 1

        # Evaluate similarity
        similarity_score = compute_semantic_similarity(
            generated_response, golden_response
        )
        print("=====================================================")
        print(f"Processed row {idx + 1}: Similarity = {similarity_score}")
        print(
            f"Generated response: {generated_response}\n"
            f"Golden response: {golden_response}\n"
        )
        print("=====================================================")

        semantic_similarities.append(similarity_score)

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        semantic_similarities.append(None)

# Add results to DataFrame
df["base_semantic_similarity"] = semantic_similarities

# Save updated DataFrame
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")
