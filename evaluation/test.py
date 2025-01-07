import json
import re


def fix_json_format(json_str):
    """
    Attempt to fix and parse a malformed JSON string.
    Handles:
    - Single quotes converted to double quotes for JSON compliance.
    - Escaping quotes inside JSON content.
    - Removing trailing commas in arrays or objects.
    """
    try:
        # Directly parse the JSON if it's already valid
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # Replace single quotes with double quotes for JSON compliance
            fixed_str = json_str.replace("'", '"')

            # Escape double quotes within values to ensure valid JSON
            fixed_str = re.sub(
                r'(?<!\\)"(?=(?:[^"]*"[^"]*")*[^"]*$)', r'\\"', fixed_str
            )

            # Remove trailing commas within objects or arrays
            fixed_str = re.sub(r",\s*([\]}])", r"\1", fixed_str)

            # Parse the corrected JSON string
            return json.loads(fixed_str)
        except json.JSONDecodeError as e:
            print(f"Failed to fix JSON: {json_str}, error: {e}")
            return None


# Example malformed JSON
conversation_history = """[{'role': 'user', 'content': 'Hello! I was hoping to schedule an appointment. Could you help me with that?'}, {'role': 'assistant', 'content': 'Hello! Thank you for calling Company X. What can I help you with today?'}, {'role': 'user', 'content': "Yes, I'm looking to schedule an appointment. Could you let me know what dates and times are available?"}, {'role': 'assistant', 'content': 'Sure, I can assist with that. Before we get started, could you please confirm both your first and last name?'}]"""

print("Before Fixing:")
print(conversation_history)

fixed_json = fix_json_format(conversation_history)

print("\nAfter Fixing:")
print(json.dumps(fixed_json, indent=2) if fixed_json else "Failed to fix JSON")
