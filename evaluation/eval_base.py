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

        # 1) Omit the last assistant message if it exists
        #    Check if last role is 'assistant'.
        last_assistant_message = None
        if convo_history and convo_history[-1]["role"] == "assistant":
            last_assistant_message = convo_history[-1]["content"]
            # remove last assistant message
            convo_history = convo_history[:-1]

        # 2) Also hide the last user message if it exists to simulate real time conversation
        # Where the  back and forth communication will be audited by the LLM and add additional information while streaming to TTS to help augement the next generation.
        # For simplicity, we will just do this statically here but the latency optimizing implementation is described in the paper.
        #    Check if last role is 'user'.
        last_user_message = None
        if convo_history and convo_history[-1]["role"] == "user":
            last_user_message = convo_history[-1]["content"]
            # remove last user message
            convo_history = convo_history[:-1]

        # We might want to identify the step from the last assistant message
        step_identifier = call_llm_to_find_step(
            convo_history[-1]["content"], convo_history, navigation_map
        )

        # 3) Augment the system prompt with a line about the step
        #    "You were at step {step_identifier}"
        system_prompt_modified = (
            f"{system_prompt}\n\n"
            f"You were at step {step_identifier} based on the latest assistant message.\n"
            "Now continue from that context."
        )

        # -- Call LLM to generate a response --
        generated_response = call_llm(
            system_prompt_modified, convo_history, last_user_message
        )
        generated_response = clean_response(generated_response)

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
