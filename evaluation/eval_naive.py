# eval_naive.py

import os
import pandas as pd
import openai
import json
from dotenv import load_dotenv
from evidently.report import Report
from evidently.metric_preset import TextEvals
from evidently import ColumnMapping
from evidently.descriptors import SemanticSimilarity

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


def evaluate_similarity(generated_response, golden_response):
    """
    Computes the semantic similarity between the generated response and the golden response.
    Returns the similarity score.
    """
    data = pd.DataFrame(
        {"generated": [generated_response], "golden": [golden_response]}
    )

    column_mapping = ColumnMapping(text_features=["generated", "golden"])

    report = Report(
        metrics=[
            TextEvals(
                column_name="generated",
                descriptors=[
                    SemanticSimilarity(
                        with_column="golden", display_name="Semantic Similarity"
                    )
                ],
            )
        ]
    )
    report.run(current_data=data, column_mapping=column_mapping)
    evaluated_data = report.datasets().current

    # Debugging: Print data structure
    print(evaluated_data.head())

    return evaluated_data["Semantic Similarity"].iloc[0]


# Load dataset
df = pd.read_csv(INPUT_FILE)
semantic_similarities = []

for idx, row in df.iterrows():
    try:
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

        generated_response = call_llm(system_prompt, convo_history, last_user_message)
        similarity_score = evaluate_similarity(generated_response, golden_response)

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
