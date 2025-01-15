import os
import pandas as pd
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
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
navigation_map = node_manager.get_navigation_map()
prompt_manager = PromptManager(node_manager)

# File paths
INPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/eval_naive.csv")

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

# Evaluation loop
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
            convo_history[1],
        ]  # Add the first user message
        generated_response = None

        # We'll keep a few states that can get updated with submap info
        current_system_prompt = prompt_manager.get()
        current_navi_map = format_flow_steps(navigation_map)
        step = 0
        i = 1
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
                if i + 1 < len(convo_history):
                    messages.append(convo_history[i + 1])  # Add the next user message

                current_step = call_llm_to_find_step(
                    turn["content"], messages, navigation_map
                )
                if current_step != -1 and current_step != "-1":
                    try:
                        step_identifier = int(current_step)
                        step = current_step
                    except Exception:
                        print("Error converting step to integer. Using previous step.")
                        step_identifier = int(step)
                last_node_type = node_manager.full_map[step_identifier]
                if "terminate" in str(last_node_type):
                    print(
                        "--------------------- Conversation ended. ---------------------"
                    )
                    break
            i += 1
        # Now, after processing all user messages, generated_response should hold
        # the *last* assistant response from the LLM.
        # Evaluate similarity
        similarity_score = compute_semantic_similarity(
            generated_response, golden_response
        )
        print("=====================================================")
        print(f"Processed row {idx + 1}: Similarity = {similarity_score}")
        print(
            f"Generated response: {generated_response} \nGolden response: {golden_response} \n"
        )
        print("=====================================================")
        semantic_similarities.append(similarity_score)

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        semantic_similarities.append(None)

# Add results to DataFrame
df["naive_semantic_similarity"] = semantic_similarities

# Save updated DataFrame
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")
