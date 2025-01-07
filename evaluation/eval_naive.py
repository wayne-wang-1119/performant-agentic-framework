import os
import pandas as pd
import ast
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load pre-trained model for semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and efficient

# File paths
INPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/eval_naive.csv")


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
    if isinstance(response, str):
        # Remove surrounding quotes if they exist
        response = response.strip('"')
        response = response.strip()
    return response


def compute_semantic_similarity(response_1, response_2):
    """
    Compute semantic similarity between two responses using cosine similarity.
    """
    embeddings = model.encode([response_1, response_2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    similarity = max(0.0, min(similarity, 1.0))
    return similarity


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
        convo_history = row["convo_history"]
        golden_response = row["golden_response"]
        convo_history = ast.literal_eval(convo_history)
        golden_response = clean_response(ast.literal_eval(golden_response))
        if convo_history[-1]["role"] == "assistant":
            convo_history = convo_history[:-1]

        last_user_message = next(
            (
                msg["content"]
                for msg in reversed(convo_history)
                if msg["role"] == "user"
            ),
            None,
        )
        if not last_user_message:
            print(f"Row {idx}: No user message found. Skipping.")
            semantic_similarities.append(None)
            continue

        # Call LLM to generate a response
        generated_response = call_llm(system_prompt, convo_history, last_user_message)
        generated_response = clean_response(generated_response)

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
df["semantic_similarity"] = semantic_similarities

# Save updated DataFrame
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")
