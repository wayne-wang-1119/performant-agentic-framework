import os
import ast
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Tuple

from openai import OpenAI
from prompt_manager import NodeManager
from prompt_manager import PromptManager
from .utils import *

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------- NodeManager & Navigation Map -------------------
# Instantiate NodeManager. Make sure your NodeManager has also been updated
# to store embeddings via OpenAI, so that node_manager.node_embeddings is populated.
node_manager = NodeManager()
navigation_map = node_manager.get_submap_upto_node(0)
prompt_manager = PromptManager(node_manager)


def find_step_with_vectors(
    assistant_message: str, current_node_id: int
) -> Tuple[int, float]:
    """
    Vectorize assistant_message and compare it to each node's embedding
    using the dot product or cosine similarity.
    Returns (best_node_id, best_score).
    """
    # Get the embedding for the message
    embedding = vectorize_prompt("text-embedding-3-small", assistant_message)

    best_node_id = None
    best_score = float("-inf")

    # Compare to each child node embedding in node_manager
    for node_id, node_emb in node_manager.node_embeddings.items():
        # Use dot product
        all_children_ids = node_manager.get_children(node_id)
        if node_id in all_children_ids:
            score = dot_product(embedding, node_emb)
            if score > best_score:
                best_score = score
                best_node_id = node_id

    # If best_score > 0.8, we consider that "good enough"; otherwise we return None
    if best_score > 0.8:
        return best_node_id, best_score
    return None, 0.0


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
        convo_history = json.loads(convo_history_str)
        golden_response = clean_response(golden_response_str)

        if not convo_history:
            print(f"Row {idx}: Empty conversation history. Skipping.")
            semantic_similarities.append(None)
            break

        messages = [
            convo_history[0],
        ]  # Add the first user message
        generated_response = None

        # We'll keep a few states that can get updated with submap info
        current_system_prompt = prompt_manager.get()
        current_navi_map = format_flow_steps(navigation_map)
        step = 0

        i = 0
        while i < len(convo_history):
            turn = messages[i]

            # If the turn is from the assistant (dataset's assistant),
            # we add it to the conversation context, then run the step-finder.
            if turn["role"] == "user":
                user_msg = turn["content"]
                assistant_reply = call_llm(current_system_prompt, messages, user_msg)
                assistant_reply = clean_response(assistant_reply)

                generated_response = assistant_reply
                messages.append({"role": "assistant", "content": assistant_reply})

                print("X" * 50)
                print("Map we are using:", current_navi_map)
                print("System prompt we are using:", current_system_prompt)
            else:
                assistant_msg = turn["content"]

                # 1) Step finder logic: vector method + LLM method, parallel both to optimize latency
                vector_step_id, vector_step_score = find_step_with_vectors(
                    assistant_msg, step
                )
                llm_step_str = call_llm_to_find_step(
                    assistant_msg, messages, current_navi_map
                )

                # 2) Decide which step to use (vector vs LLM)
                if vector_step_id is not None:
                    current_step = str(vector_step_id)
                    print(
                        "Using vector method for step:",
                        vector_step_id,
                        vector_step_score,
                    )
                else:
                    current_step = llm_step_str
                    print("Using LLM method for step:", llm_step_str)

                if current_step != -1 and current_step != "-1":
                    try:
                        step_identifier = int(current_step)
                        step = current_step
                    except Exception:
                        print("Error converting step to integer. Using previous step.")
                        step_identifier = int(step)

                # 4) Build submap and update system prompt
                submap = node_manager.get_submap_upto_node(step_identifier)
                current_navi_map = format_flow_steps(submap)
                current_system_prompt = (
                    f"{current_system_prompt}\n\n"
                    f"You were at step {step} based on the latest assistant message.\n"
                    f"Below is a partial navigation map relevant to your current step:\n{current_navi_map}\n\n"
                    "Now continue from that context."
                )
                if i + 1 < len(convo_history):
                    messages.append(convo_history[i + 1])  # Add the next user message

                last_node_type = node_manager.full_map[step_identifier]
                if "terminate" in str(last_node_type):
                    print(
                        "--------------------- Conversation ended. ---------------------"
                    )
                    break

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
        break

# 9) Write results
df["optimized_semantic_similarity"] = semantic_similarities
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")
