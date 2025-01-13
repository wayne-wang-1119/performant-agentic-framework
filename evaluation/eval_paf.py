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
from utils import *

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------- NodeManager & Navigation Map -------------------
# Instantiate NodeManager. Make sure your NodeManager has also been updated
# to store embeddings via OpenAI, so that node_manager.node_embeddings is populated.
node_manager = NodeManager()
navigation_map = node_manager.get_submap_upto_node(0)
print("0 Navigation Map:", navigation_map)


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
