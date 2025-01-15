import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


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
                    "caller asks about other services like installation": 5,
                    "caller wants to end the call": 10,
                    # New routes from the start node
                    "caller wants to inquire about billing": 11,
                    "caller asking technical support with existing product they have": 12,
                },
            },
            1: {
                "instruction": "Before we get started, could you please confirm both your first and last name?",
                "navigation": {
                    "caller confirms name": 2,
                    # Additional branch if caller has a question about privacy
                    "caller asks about privacy policy or does not give name": 13,
                },
            },
            2: {
                "instruction": "Thanks for that. Can you give me your phone number, please?",
                "navigation": {
                    "caller provides number": 3,
                    # Option to skip providing number
                    "caller hesitates to provide number": 14,
                },
            },
            3: {
                "instruction": "I have your phone number as [number]. Is that the best number to reach you?",
                "navigation": {
                    "caller confirms number": 4,
                    # Option to provide an alternative number
                    "caller want to change number on profile": 15,
                },
            },
            4: {
                "instruction": "Now, could I please have your full address? Please include the street, city, state, and zip code.",
                "navigation": {
                    "caller provides address": 6,
                    # Option if caller doesn't know the full address
                    "caller unsure about address": 16,
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
                    # Option if caller wants to reschedule an existing appointment
                    "caller wants to reschedule": 17,
                },
            },
            7: {
                "instruction": "Got it. One moment while I pull available times for that day.",
                "navigation": {
                    "caller selects a time to schedule": 8,
                    # Option if caller wants to know more about services on that day
                    "caller asks about service details": 18,
                },
            },
            8: {
                "instruction": "Perfect! I have booked your appointment for [date and time]. Is there anything else I can assist you with?",
                "navigation": {
                    "caller has no more questions": 9,
                    "caller wants to go over the schedule again": 11,  # Looping back to billing for additional query
                    # New branch for additional service request after booking
                    "caller wants to add another service": 19,
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
            # New nodes for expanded paths
            11: {
                "instruction": "You've reached the billing department. How can we assist you with your billing question?",
                "navigation": {
                    "caller asks about an invoice": 20,
                    "caller wants to make a payment": 21,
                    "caller dispute a charge": 22,
                    "caller wants to go back to main menu": 0,
                },
            },
            12: {
                "instruction": "Technical support here. Can you describe the issue you're experiencing?",
                "navigation": {
                    "caller describes internet issues": 23,
                    "caller describes billing software issue": 24,
                    "caller wants to speak to a specialist": 25,
                    "caller wants to go back to main menu": 0,
                },
            },
            13: {
                "instruction": "Our privacy policy ensures your data is safe. Do you have any specific questions?",
                "navigation": {
                    "caller asks about data storage": 26,
                    "caller asks about data sharing": 27,
                    "caller wants to go back": 1,
                },
            },
            14: {
                "instruction": "I understand. Would you like to continue without providing a phone number?",
                "navigation": {
                    "caller agrees to continue without number": 4,
                    "caller changes mind and provides number": 3,
                },
            },
            15: {
                "instruction": "Thanks for providing the alternative number. I'll update our records. Is that correct?",
                "navigation": {
                    "caller confirms update": 4,
                    "caller wants to change it again": 15,
                },
            },
            16: {
                "instruction": "No worries. Please provide as much of your address as you can.",
                "navigation": {
                    "caller provides partial address": 6,
                    "caller cannot provide address": 28,
                },
            },
            17: {
                "instruction": "Sure, let's reschedule your appointment. What new day works for you?",
                "navigation": {
                    "caller provides new day": 7,
                    "caller wants to cancel appointment instead": 29,
                },
            },
            18: {
                "instruction": "We offer several services on that day. Would you like details on any particular service?",
                "navigation": {
                    "caller asks about service A": 30,
                    "caller asks about service B": 31,
                    "caller wants to speak to a service specialist": 32,
                    "caller wants to go back": 7,
                },
            },
            19: {
                "instruction": "Sure, what additional service would you like to add?",
                "navigation": {
                    "caller wants service X": 33,
                    "caller wants service Y": 34,
                    "caller wants to add nothing else": 9,
                },
            },
            # Additional new nodes beyond 19
            20: {
                "instruction": "For invoice questions, please provide your invoice number.",
                "navigation": {
                    "caller provides invoice number": 35,
                    "caller doesn't have invoice number": 36,
                },
            },
            21: {
                "instruction": "To make a payment, please choose your payment method.",
                "navigation": {
                    "caller chooses credit card": 37,
                    "caller chooses paypal": 38,
                    "caller wants to go back": 11,
                },
            },
            22: {
                "instruction": "I'm sorry to hear about the charge dispute. Could you provide more details?",
                "navigation": {
                    "caller provides dispute details": 39,
                    "caller wants to speak to a manager": 40,
                },
            },
            23: {
                "instruction": "I'm sorry you're having internet issues. Let's run a quick diagnostic.",
                "navigation": {
                    "caller agrees to diagnostic": 41,
                    "caller wants to speak to an engineer": 42,
                },
            },
            24: {
                "instruction": "For billing software issues, can you specify the problem?",
                "navigation": {
                    "caller describes software issue": 43,
                    "caller wants to schedule a support call": 44,
                },
            },
            25: {
                "instruction": "Connecting you to a specialist. Please hold.",
                "navigation": "terminate by specialist transfer",
            },
            # Further nodes can be continued similarly...
            26: {
                "instruction": "Our data is stored on secure servers. Your information is encrypted.",
                "navigation": {
                    "caller has more questions": 13,
                    "caller wants to go back": 1,
                },
            },
            27: {
                "instruction": "We only share data with your consent or legal requirements.",
                "navigation": {
                    "caller has more questions": 13,
                    "caller wants to go back": 1,
                },
            },
            28: {
                "instruction": "Without your address, we might face difficulties scheduling. Can you provide at least your city and state?",
                "navigation": {
                    "caller provides partial address": 6,
                    "caller wants to cancel": 29,
                },
            },
            29: {
                "instruction": "Your appointment has been cancelled. Is there anything else I can assist you with?",
                "navigation": {
                    "caller wants to book a new appointment": 1,
                    "caller wants to speak to someone": 5,
                    "caller has no more questions": 9,
                },
            },
            30: {
                "instruction": "Service A includes... Would you like to schedule this service?",
                "navigation": {
                    "caller wants to schedule service A": 1,
                    "caller wants more information": 18,
                },
            },
            31: {
                "instruction": "Service B includes... Would you like to schedule this service?",
                "navigation": {
                    "caller wants to schedule service B": 1,
                    "caller wants more information": 18,
                },
            },
            32: {
                "instruction": "Connecting you to a service specialist. Please hold.",
                "navigation": "terminate by specialist transfer",
            },
            33: {
                "instruction": "Service X added to your appointment. Anything else?",
                "navigation": {
                    "caller wants another service": 19,
                    "caller is done": 9,
                },
            },
            34: {
                "instruction": "Service Y added to your appointment. Anything else?",
                "navigation": {
                    "caller wants another service": 19,
                    "caller is done": 9,
                },
            },
            35: {
                "instruction": "Thank you. One moment while I look up your invoice.",
                "navigation": {
                    "caller waiting": 20,  # Loop or additional logic for waiting...
                },
            },
            36: {
                "instruction": "No problem. How else may I assist you with billing?",
                "navigation": {
                    "caller asks about payment options": 21,
                    "caller wants to go back": 11,
                },
            },
            37: {
                "instruction": "Processing credit card payment. Please hold.",
                "navigation": {
                    "caller payment successful": 9,
                    "caller payment failed": 22,
                },
            },
            38: {
                "instruction": "Processing PayPal payment. Please hold.",
                "navigation": {
                    "caller payment successful": 9,
                    "caller payment failed": 22,
                },
            },
            39: {
                "instruction": "I will file a dispute report for you. Do you want a confirmation email?",
                "navigation": {
                    "caller wants email confirmation": 26,
                    "caller declines": 9,
                },
            },
            40: {
                "instruction": "Connecting you to a manager. Please hold.",
                "navigation": "terminate by manager transfer",
            },
            41: {
                "instruction": "Running diagnostics... Please wait.",
                "navigation": {
                    "caller reports issue resolved": 9,
                    "caller still has problems": 42,
                },
            },
            42: {
                "instruction": "I'll connect you to an engineer for further assistance.",
                "navigation": "terminate by engineer transfer",
            },
            43: {
                "instruction": "For software issues, try restarting the application. Did that solve the problem?",
                "navigation": {
                    "caller yes solved": 9,
                    "caller still experiencing issues": 44,
                },
            },
            44: {
                "instruction": "Let's schedule a support call to dive deeper into the issue.",
                "navigation": {
                    "caller provides availability": 45,
                },
            },
            45: {
                "instruction": "Thank you. Your support call is scheduled. Anything else I can help you with?",
                "navigation": {
                    "caller no further assistance needed": 9,
                    "caller has another issue": 12,
                },
            },
            # ... continue adding nodes as needed to further expand the tree.
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
