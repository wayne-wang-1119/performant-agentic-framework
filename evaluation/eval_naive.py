# eval_naive.py

import os
import pandas as pd
import openai
import json
from dotenv import load_dotenv
from evidently.test_suite import TestSuite
from evidently.descriptors import *
from evidently.tests import *

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

    response = openai.ChatCompletion.create(
        model="gpt-4", messages=messages  # Ensure correct model name
    )
    return response["choices"][0]["message"]["content"]


def evaluate_with_evidently(generated_responses, golden_responses):
    """
    Computes semantic similarity and regression metrics using Evidently's TestSuite.
    Returns a dictionary with evaluation results.
    """
    # Prepare the data
    data = pd.DataFrame(
        {
            "response": golden_responses,
            "generated_response": generated_responses,
        }
    )

    # Create a TestSuite for semantic similarity
    test_suite = TestSuite(
        tests=[
            TestColumnValueMean(
                column_name=SemanticSimilarity(display_name="Semantic Similarity").on(
                    ["response", "generated_response"]
                ),
                gte=0.0,  # No threshold; just compute the value
            )
        ]
    )

    # Run the test suite
    test_suite.run(reference_data=None, current_data=data)

    # Extract results as JSON and parse them
    test_results = test_suite.json()
    similarity_score = test_results["tests"][0]["parameters"]["value"]

    return similarity_score


# Load dataset
df = pd.read_csv(INPUT_FILE)
semantic_similarities = []

for idx, row in df.iterrows():
    try:
        # Extract system prompt, conversation history, and golden response
        system_prompt = row["system_prompt"]
        convo_history = json.loads(row["convo_history"])
        golden_response = row["golden_response"]

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

        # Call the LLM to get the generated response
        generated_response = call_llm(system_prompt, convo_history, last_user_message)

        # Evaluate similarity using Evidently
        similarity_score = evaluate_with_evidently(
            [generated_response], [golden_response]
        )
        print(f"Processed row {idx + 1}: Similarity = {similarity_score}")

        semantic_similarities.append(similarity_score)

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        semantic_similarities.append(None)

# Add results to DataFrame
df["semantic_similarity"] = semantic_similarities

# Save updated DataFrame
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved results to {OUTPUT_FILE}")
