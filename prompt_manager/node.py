import os
import json
import requests
from typing import List, Dict, Any


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


class NodeManager:
    def __init__(self):
        """
        Initialize the NodeManager and load all of your flow-map nodes.
        Instead of using SentenceTransformer, we call `vectorize_prompt`
        for each instruction to get its embedding from OpenAI.
        """
        self.full_map: Dict[int, Dict[str, Any]] = {
            0: {
                "instruction": "Hello! Thank you for calling Company X. What can I help you with today?",
                "navigation": {
                    "caller wants to book an appointment": 1,
                    "caller asks about other_services": 5,
                    "caller wants to end the call": 10,
                },
            },
            1: {
                "instruction": "Before we get started, could you please confirm both your first and last name?",
                "navigation": {
                    "caller confirm_name": 2,
                },
            },
            2: {
                "instruction": "Thanks for that. Can you give me your phone number, please?",
                "navigation": {
                    "caller provids number": 3,
                },
            },
            3: {
                "instruction": "I have your phone number as [number]. Is that the best number to reach you?",
                "navigation": {
                    "caller confirm_number": 4,
                },
            },
            4: {
                "instruction": "Now, could I please have your full address? Please include the street, city, state, and zip code.",
                "navigation": {
                    "caller provides address": 6,
                },
            },
            5: {
                "instruction": "Okay, let me transfer you. One moment, please.",
                "navigation": "terminate the call by transfer to another agent",
            },
            6: {
                "instruction": "Thank you. I have noted your address. What day are you looking for us to come out?",
                "navigation": {
                    "caller provides day to schedule": 7,
                },
            },
            7: {
                "instruction": "Got it. One moment while I pull available times for that day.",
                "navigation": {
                    "caller selects a time to schedule": 8,
                },
            },
            8: {
                "instruction": "Perfect! I have booked your appointment for [date and time]. Is there anything else I can assist you with?",
                "navigation": {
                    "caller has no more quesitons": 9,
                },
            },
            9: {
                "instruction": "Have a great day!",
                "navigation": "terminate",
            },
            10: {
                "instruction": "Goodbye! Have a great day!",
                "navigation": "terminate",
            },
            11: {
                "instruction": "I can schedule that for you. What day are you looking for us to come out?",
                "navigation": {
                    "callers provides day": 7,
                    "caller has general questions": 13,
                },
            },
            12: {
                "instruction": "I would love to help answer that question, but I can only schedule appointments. Would you like to schedule an appointment?",
                "navigation": {
                    "caller wants to book appointment": 1,
                    "caller has reserves about other services": 5,
                    "caller wants to end the call or is frustrated and off topic": 10,
                    "caller has some specific inquiries": 15,
                },
            },
            13: {
                "instruction": "We can re-schedule your appointment to next month or next year. Which do you prefer?",
                "navigation": {
                    "caller wants next month": 7,
                    "caller wants next year": 12,
                    "caller wants to go back to scheduling": 0,
                },
            },
            14: {
                "instruction": "Confirming next-year scheduling. You might lose your current slot. Continue?",
                "navigation": {
                    "caller wants to schedule next year": 8,
                    "caller wants to schedule the appointment again": 9,
                    "caller has questions about scheduling": 15,
                },
            },
            15: {
                "instruction": "Your current service request might require shipping a device or scheduling on-site. Which do you want?",
                "navigation": {
                    "ship_device": 16,
                    "on_site": 1,
                    "off_path": 12,
                },
            },
            16: {
                "instruction": "We see you opted for shipping. This might conflict with normal on-site scheduling. Proceed anyway?",
                "navigation": {
                    "proceed_shipping": 17,
                    "cancel_shipping": 9,
                },
            },
            17: {
                "instruction": "We can expedite shipping if you confirm your address again or speak with general support.",
                "navigation": {
                    "confirm_address_again": 6,
                    "general_support": 5,
                },
            },
            18: {
                "instruction": "We found an alternative scheduling queue. It can handle your request faster, but might override your current appointment.",
                "navigation": {
                    "override_appt": 7,
                    "stay_in_queue": 9,
                    "redirect_to_12": 12,
                },
            },
            19: {
                "instruction": "We can create multiple appointments if you’d like. This is a new feature. Confirm or revert?",
                "navigation": {
                    "confirm_multi": 8,
                    "revert_main": 0,
                },
            },
            20: {
                "instruction": "We see conflicting data about shipping vs. on-site visits. Clarify your preference. Our system gets confused otherwise.",
                "navigation": {
                    "prefer_shipping": 16,
                    "prefer_on_site": 1,
                    "stop_now": 10,
                },
            },
            21: {
                "instruction": "Need a courtesy hold on your appointment date? This might pause everything. Are you sure?",
                "navigation": {
                    "hold_appointment": 22,
                    "no_hold": 9,
                },
            },
            22: {
                "instruction": "Courtesy hold set. We cannot proceed until you confirm again. Confirm or end?",
                "navigation": {
                    "confirm_again": 8,
                    "end_call": 10,
                },
            },
            23: {
                "instruction": "We can escalate your call to a specialized scheduling queue for complex requests. This might remove prior data. Proceed?",
                "navigation": {
                    "yes_escalate": 24,
                    "no_regular": 9,
                },
            },
            24: {
                "instruction": "Escalation complete. This queue handles re-scheduling and shipping at once. Provide a day?",
                "navigation": {
                    "provide_day": 7,
                    "abort": 10,
                },
            },
            25: {
                "instruction": "We suspect you want a weekend appointment, which is outside normal hours. Should we forcibly schedule it anyway?",
                "navigation": {
                    "yes_force": 8,
                    "no_return": 9,
                    "maybe_confusion": 20,
                },
            },
            26: {
                "instruction": "We can gather multiple addresses for a single appointment. This might break our system. Confirm or revert?",
                "navigation": {
                    "confirm_multi_addr": 27,
                    "revert": 9,
                },
            },
            27: {
                "instruction": "Multiple addresses added. Are you sure we have them correct? If not, shipping or scheduling might fail.",
                "navigation": {
                    "correct_addr": 6,
                    "ignore_mismatch": 8,
                },
            },
            28: {
                "instruction": "We found an attempt to schedule in the past. Do you want a retroactive schedule or correct the date?",
                "navigation": {
                    "retroactive": 29,
                    "correct_date": 7,
                },
            },
            29: {
                "instruction": "Retroactive scheduling is highly unusual and may lead to system errors. Continue anyway?",
                "navigation": {
                    "continue_retro": 9,
                    "end_call": 10,
                },
            },
            30: {
                "instruction": "You’re requesting returns or exchanges. Normally, we only handle appointments. Try forcing it or revert?",
                "navigation": {
                    "force_returns": 31,
                    "revert_normal": 8,
                },
            },
            31: {
                "instruction": "Forcing returns. This might break the scheduling flow. We can finalize or revert now.",
                "navigation": {
                    "finalize_return": 9,
                    "revert_flow": 1,
                },
            },
            32: {
                "instruction": "We can combine shipping, scheduling, and returns in one request, but that often leads to confusion. Are you absolutely sure?",
                "navigation": {
                    "combine_all": 33,
                    "no_combine": 9,
                },
            },
            33: {
                "instruction": "All options combined! This is extremely error-prone. Should we proceed or go back?",
                "navigation": {
                    "proceed_combo": 8,
                    "back_to_main": 0,
                },
            },
            34: {
                "instruction": "What can I help you with today? I can assist with scheduling appointments or provide information on our services.",
                "navigation": {
                    "confirm": 8,
                    "does_not_want_schedule": 0,
                },
            },
            35: {
                "instruction": "Interesting to hear! Can you tell me more about what you're looking for?",
                "navigation": {
                    "provide_info": 36,
                },
            },
            36: {
                "instruction": "I see. Let me check if we have that available. One moment, please.",
                "navigation": {
                    "check_availability": 0,
                },
            },
        }

        # Build a dictionary of embeddings for each node
        self.node_embeddings: Dict[int, List[float]] = {}
        for node_id, node_data in self.full_map.items():
            instruction_text = node_data["instruction"]
            # Call the OpenAI embedding endpoint for each instruction
            embedding = vectorize_prompt("text-embedding-3-small", instruction_text)
            self.node_embeddings[node_id] = embedding

    def get_navigation_map(self, nodes=None) -> Dict[int, Dict[str, Any]]:
        """
        Returns a subset of the navigation map based on the given node list.
        If nodes is None or empty, the full map is returned.
        """
        if not nodes:
            return self.full_map

        # Validate the provided nodes
        invalid_nodes = [node for node in nodes if node not in self.full_map]
        if invalid_nodes:
            raise ValueError(f"Invalid nodes: {invalid_nodes}")

        # Return the filtered map
        return {node: self.full_map[node] for node in nodes}

    def get_submap_upto_node(self, target_node: int) -> Dict[int, Dict[str, Any]]:
        """
        Build a 'path' from node 0 to 'target_node' (if possible),
        then collect:
          - all nodes on that path
          - the children of each node on that path (if they exist)
        Returns a sub-map (dict) that you can pass to your LLM.
        """
        visited = set()
        path = []

        def dfs(current, goal, current_path):
            if current in visited:
                return False
            visited.add(current)
            current_path.append(current)

            if current == goal:
                return True

            node_data = self.full_map.get(current, {})
            nav = node_data.get("navigation", {})

            # If navigation is a dict, explore children
            if isinstance(nav, dict):
                for child_node in nav.values():
                    if isinstance(child_node, int):
                        if dfs(child_node, goal, current_path):
                            return True

            current_path.pop()
            return False

        found = dfs(0, target_node, path)
        if not found:
            # If there's no path from 0 to target_node,
            # just return a single-node map if it exists
            return (
                {target_node: self.full_map[target_node]}
                if target_node in self.full_map
                else {}
            )

        # Build submap from the nodes in 'path' + the children of those nodes
        submap_nodes = set(path)

        for node_id in path:
            navigation = self.full_map[node_id].get("navigation", {})
            if isinstance(navigation, dict):
                for child in navigation.values():
                    if isinstance(child, int):
                        submap_nodes.add(child)

        # Construct the dictionary
        filtered_map = {}
        for node_id in submap_nodes:
            filtered_map[node_id] = self.full_map[node_id]

        return filtered_map
