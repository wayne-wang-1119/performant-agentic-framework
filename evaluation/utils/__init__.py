import math
import os
import re
import ast
import json
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List, Tuple

from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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

    # Return the first embedding
    embedding = data["data"][0]["embedding"]
    return embedding


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
    """
    Clean up quotes and whitespace from a string.
    """
    if isinstance(response, str):
        response = response.strip('"')
        response = response.strip()
    else:
        response = str(response)
        response = response.strip('"')
        response = response.strip()
    return response


def sanitize_string_literal(s):
    """
    Replace problematic quotes and characters with safer alternatives.
    """
    if isinstance(s, str):
        s = s.replace("’", "'").replace("‘", "'")
        s = s.replace(""", '"').replace(""", '"')
        s = clean_response(s)
    return s


def safe_literal_eval(val):
    import ast

    try:
        sanitized_val = sanitize_string_literal(val)
        return ast.literal_eval(sanitized_val)
    except Exception as e:
        print(f"Error parsing value: {val}\nError: {e}")
        return None


def compute_semantic_similarity(response_1, response_2):
    """
    Compute semantic similarity between two responses using cosine similarity
    via OpenAI embeddings.
    """
    # Get embeddings from OpenAI
    emb1 = vectorize_prompt("text-embedding-3-small", response_1)
    emb2 = vectorize_prompt("text-embedding-3-small", response_2)
    similarity = cosine_similarity(emb1, emb2)
    return similarity


def call_llm_to_find_node(assistant_message, conversation_history, navigation_map):
    """
    Call the LLM to find the node or the Node ID that the agent is on currently
    based on the latest assistant message and conversation history.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are identifying which Node ID or node number most closely "
                "aligns with the latest assistant message. Return only the digit "
                "that represents the Node/Node ID."
            ),
        },
        {
            "role": "system",
            "content": """`
### Task Instructions:
You must identify the node from the navigation map that is **most similar** to the assistant's last response. The similarity should be based on the following criteria, in order of priority:
1. **Intent**: Match the primary purpose or action of the assistant's last response (e.g., confirming a name, providing an explanation, asking for information, etc.).
2. **Key Phrases**: Look for specific keywords or actions mentioned in the assistant's last response (e.g., "confirm," "name," "phone number").
3. **Context Alignment**: Consider how the assistant's last response aligns with the expected outcomes or instructions for each node in the navigation map.

### Steps to Determine the Most Similar Node:
1. **Understand the Assistant's Intent**: Analyze the assistant's last response to identify what action it is performing (e.g., confirming a name, asking for details, etc.).
2. **Analyze the Navigation Map**: Compare the intent and key phrases of the assistant's response with the instructions and expected behaviors for each node in the navigation map.
3. **Choose the Closest Match**: Select the node that most closely matches the intent and key phrases of the assistant's response. 

If multiple nodes are similar, select the one with the closest **intent match**. Return the node number in JSON format.

### Additional Notes:
- For nodes that contain instructions that end the call or indicate ending the call (e.g. "Ok, goodbye for now"), treat them with extra caution when selecting as a response. Since these nodes end the call, they should typically appear only once. When you are evaluating potential next nodes to return, avoid prematurely ending the call.
- If the conversation is not advancing to any appropriate node, return -1.
- When you return a node that is the end call node which has instruction to end the call, you should only return that node if the latest assistant message is clearly the same as the end call message. 
- You should never return a node with instructions that do not resemble what the latest assistant message tries to achieve.

---

### Task:
Based on the navigation map, return the node that is most similar to what the AI assistant responded with in the last AI Message.
If the conversation is not advancing to any appropriate node, return -1. 
You should try to advance the conversation based on the latest assistant message.
""",
        },
        {
            "role": "system",
            "content": f"Here is the latest assistant message: {assistant_message}",
        },
        {
            "role": "system",
            "content": f"Here is the navigation map: {navigation_map}",
        },
    ]
    messages.extend(conversation_history)
    print("last assistant message:", assistant_message)
    print("conversation history:", conversation_history)
    print("navigation map:", navigation_map)

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    node_str = response.choices[0].message.content
    print("LLM response to find node:", node_str)
    return node_str


def format_flow_nodes(flow_map):
    """
    Given a dictionary of nodes (flow_map), return a string describing
    each node, its instruction, and its navigation options.
    """
    lines = []
    for node_number, node_info in flow_map.items():
        instruction = node_info.get("instruction", "No instruction found")
        line = f"On node {node_number} you have instruction '{instruction}'."

        navigation = node_info.get("navigation")
        if isinstance(navigation, dict):
            for condition, next_node in navigation.items():
                line += f" Based on condition '{condition}', you can go to node {next_node}."
        elif isinstance(navigation, str):
            line += f" Navigation action: {navigation}."
        # else: no valid navigation - skip
        lines.append(line)
    return "\n".join(lines)


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors (lists of floats).
    Returns a float in the range [0.0, 1.0] (or 0.0 if vectors are empty or mismatched).
    """
    if len(vec1) != len(vec2) or len(vec1) == 0:
        return 0.0

    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for x, y in zip(vec1, vec2):
        dot_product += x * y
        norm1 += x * x
        norm2 += y * y

    norm1 = math.sqrt(norm1)
    norm2 = math.sqrt(norm2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


def dot_product(vec1, vec2):
    """
    Compute the dot product of two vectors (lists of floats).
    Returns a float (or 0.0 if vectors are empty or mismatched).
    """
    if len(vec1) != len(vec2) or len(vec1) == 0:
        return 0.0

    result = 0.0
    for x, y in zip(vec1, vec2):
        result += x * y

    return result
