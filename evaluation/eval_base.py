import os
import json
import ast
import pandas as pd
from dotenv import load_dotenv
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

# File paths
INPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/eval_base.csv")

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
        current_navi_map = format_flow_nodes(navigation_map)
        node = 0

        i = 0
        while i < len(convo_history):
            turn = messages[i]

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
                current_node = call_llm_to_find_node(
                    turn["content"], messages, navigation_map
                )
                if current_node != -1 and current_node != "-1":
                    try:
                        node_identifier = int(current_node)
                        node = current_node
                    except Exception:
                        print("Error converting node to integer. Using previous node.")
                        node_identifier = int(node)

                # 4) Build submap and update system prompt
                submap = node_manager.get_submap_upto_node(node_identifier)
                current_navi_map = format_flow_nodes(submap)
                current_system_prompt = (
                    f"{current_system_prompt}\n\n"
                    f"You were at node {node} based on the latest assistant message.\n"
                    f"Below is a partial navigation map relevant to your current node:\n{current_navi_map}\n\n"
                    "Now continue from that context."
                )
                if i + 1 < len(convo_history):
                    messages.append(convo_history[i + 1])  # Add the next user message

                last_node_type = node_manager.full_map[node_identifier]
                if "terminate" in str(last_node_type):
                    print(
                        "--------------------- Conversation ended. ---------------------"
                    )
                    break
            i += 1

        # Evaluate similarity
        similarity_score = compute_semantic_similarity(
            generated_response, golden_response
        )
        print("=====================================================")
        print(f"Processed row {idx}: Similarity = {similarity_score}")
        print(
            f"Generated response: {generated_response}\nGolden response: {golden_response}\n"
        )
        print("=====================================================")

        semantic_similarities.append(similarity_score)

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        break

# Add results to DataFrame
df["base_semantic_similarity"] = semantic_similarities

# Save updated DataFrame
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")
