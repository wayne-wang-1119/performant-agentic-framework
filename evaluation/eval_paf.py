import os
import numpy as np
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

# Instantiate NodeManager
node_manager = NodeManager()
# still get the full map, but we'll also call get_submap_upto_node(...) later
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
            "content": f"Here is the latest assistant message: {assistant_message}",
        },
        {
            "role": "system",
            "content": f"Here is the navigation map: {navigation_map}",
        },
    ]
    messages.extend(conversation_history)

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    step_str = response.choices[0].message.content
    print("LLM response to find step:", step_str)
    return step_str


def find_step_with_vectors(assistant_message: str) -> (int, float):
    """
    Vectorize assistant_message and compare it to each node's embedding
    using the *dot product* rather than cosine similarity.

    Returns (best_node_id, best_score).
    """
    embedding = node_manager.model.encode(assistant_message)
    best_node_id = None
    best_score = float("-inf")

    for node_id, node_emb in node_manager.node_embeddings.items():
        dot_val = np.dot(embedding, node_emb)  # Dot product
        if dot_val > best_score:
            best_score = dot_val
            best_node_id = node_id

    # If best_score > 0.3, we consider that "good enough"; otherwise we return None
    if best_score > 0.3:
        return best_node_id, best_score
    return None, 0.0


# File paths
INPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/eval_paf.csv")

# Load dataset
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

        # 1) Omit last assistant message if it exists
        if convo_history[-1]["role"] == "assistant":
            convo_history.pop()

        # 2) Omit last user message if it exists
        last_user_message = None
        if convo_history and convo_history[-1]["role"] == "user":
            last_user_message = convo_history[-1]["content"]
            convo_history.pop()

        if not last_user_message:
            print(f"Row {idx}: No last user message. Skipping.")
            semantic_similarities.append(None)
            continue

        # 3) Parallel: Call vector method & LLM method to find the step
        assistant_msg_for_step = convo_history[-1]["content"]
        vector_step_id, vector_step_score = find_step_with_vectors(
            assistant_msg_for_step
        )

        llm_step_str = call_llm_to_find_step(
            assistant_msg_for_step, convo_history, navigation_map
        )

        # If vector_step_id is not None, it passed the threshold; otherwise use llm_step_str
        if vector_step_id is not None:
            print(
                "Using vector method to find step.", vector_step_id, vector_step_score
            )
            step_str = str(vector_step_id)
        else:
            print("Using LLM method to find step.", llm_step_str)
            step_str = llm_step_str

        # 4) Convert step_str to integer if possible
        try:
            step_identifier = int(re.findall(r"\d+", step_str)[0])
        except Exception:
            step_identifier = 0  # fallback, or skip row

        # 5) Build a sub-map with the path from node 0 to step_identifier
        #    plus the children of each node in that path
        submap = node_manager.get_submap_upto_node(step_identifier)

        # 6) Augment system prompt
        system_prompt_modified = (
            f"{system_prompt}\n\n"
            f"You were at step {step_identifier} based on the latest assistant message.\n"
            f"Below is a partial navigation map relevant to your current step:\n{submap}\n\n"
            "Now continue from that context."
        )

        # 7) Generate a new assistant response
        generated_response = call_llm(
            system_prompt_modified,
            convo_history,
            last_user_message,
        )
        generated_response = clean_response(generated_response)

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
