import os
import pandas as pd
import json
import requests
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
model = SentenceTransformer("all-MiniLM-L6-v2")

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
        response = response.strip('"')
        response = response.strip()
    return response


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

    # The embedding is typically in the first entry of data["data"]
    embedding = data["data"][0]["embedding"]
    return embedding


def compute_semantic_similarity(response_1, response_2):
    """
    Compute semantic similarity between two responses using cosine similarity
    via OpenAI embeddings.
    """
    # Get the embeddings from OpenAI
    embedding_1 = vectorize_prompt("text-embedding-3-small", response_1)
    embedding_2 = vectorize_prompt("text-embedding-3-small", response_2)

    # Use sklearn's cosine_similarity
    similarity = cosine_similarity([embedding_1], [embedding_2])[0][0]
    # Clamp the similarity to [0.0, 1.0]
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
        messages = [convo_history[0]]  # Add the first assistant message
        generated_response = None
        i = 0
        while i < len(convo_history):
            turn = convo_history[i]

            if turn["role"] == "user":
                # 1) Add the user message
                user_msg = turn["content"]
                messages.append({"role": "user", "content": user_msg})

                # 2) Call LLM to get an assistant response
                assistant_reply = call_llm(system_prompt, messages, user_msg)
                assistant_reply = clean_response(assistant_reply)

                # 3) Keep track of the latest generated assistant response
                generated_response = assistant_reply

                # 4) Append that assistant reply into the messages
                messages.append({"role": "assistant", "content": assistant_reply})

            else:
                # For "assistant" messages in the dataset, we do NOT call the LLM.
                pass

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
