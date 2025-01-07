import os
import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv
from evidently.test_suite import TestSuite
from evidently.descriptors import SemanticSimilarity
from evidently.tests import TestColumnValueMean

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# File paths
INPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/eval_naive.csv")


# Fix JSON formatting issues
def fix_json_format(json_str):
    """
    Fix JSON formatting issues:
    - Replace single quotes with double quotes
    - Escape any problematic characters
    """
    try:
        return json.loads(json_str)  # Attempt direct parsing
    except json.JSONDecodeError:
        try:
            # Replace single quotes with double quotes
            fixed_str = json_str.replace("'", '"')
            return json.loads(fixed_str)
        except json.JSONDecodeError as e:
            print(f"Failed to fix JSON: {json_str}, error: {e}")
            return None  # Return None if completely invalid


# Call the LLM
def call_llm(system_prompt, conversation_history, user_message):
    """
    Calls the LLM with the system prompt, conversation history, and user message.
    Returns the generated assistant response.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages  # Use your preferred model)
    return response.choices[0].message.content


# Evaluate similarity using Evidently
def evaluate_with_evidently(generated_responses, golden_responses):
    """
    Computes semantic similarity using Evidently's TestSuite.
    Returns the similarity score.
    """
    data = pd.DataFrame(
        {
            "response": golden_responses,
            "generated_response": generated_responses,
        }
    )

    test_suite = TestSuite(
        tests=[
            TestColumnValueMean(
                column_name=SemanticSimilarity(display_name="Semantic Similarity").on(
                    ["response", "generated_response"]
                ),
                gte=0.0,  # Compute value without a threshold
            )
        ]
    )

    test_suite.run(reference_data=None, current_data=data)
    test_results = test_suite.json()
    return test_results["tests"][0]["parameters"]["value"]


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

# Fix JSON formatting in the 'convo_history' column
df["convo_history"] = df["convo_history"].apply(fix_json_format)

# Evaluation loop
semantic_similarities = []
for idx, row in df.iterrows():
    try:
        system_prompt = row["system_prompt"]
        convo_history = row["convo_history"]
        golden_response = row["golden_response"]

        if not convo_history or not isinstance(convo_history, list):
            raise ValueError(f"Invalid conversation history at row {idx}")

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

        # Evaluate similarity
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
