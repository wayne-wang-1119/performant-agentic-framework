from sentence_transformers import SentenceTransformer


class NodeManager:
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # Using OpenAI's latest vector model v3 small is also performant and optimizes the latency
        # Define the full map of nodes
        self.full_map = {
            0: {
                "instruction": "Hello! Thank you for calling Company X. What can I help you with today?",
                "navigation": {
                    "book_appointment": 1,
                    "other_services": 5,
                    "end_call": 10,
                },
            },
            1: {
                "instruction": "Before we get started, could you please confirm both your first and last name?",
                "navigation": {
                    "confirm_name": 2,
                },
            },
            2: {
                "instruction": "Thanks for that. Can you give me your phone number, please?",
                "navigation": {
                    "provide_number": 3,
                },
            },
            3: {
                "instruction": "I have your phone number as [number]. Is that the best number to reach you?",
                "navigation": {
                    "confirm_number": 4,
                },
            },
            4: {
                "instruction": "Now, could I please have your full address? Please include the street, city, state, and zip code.",
                "navigation": {
                    "provide_address": 6,
                },
            },
            5: {
                "instruction": "Okay, let me transfer you. One moment, please.",
                "navigation": "terminate",
            },
            6: {
                "instruction": "Thank you. I have noted your address. What day are you looking for us to come out?",
                "navigation": {
                    "provide_day": 7,
                },
            },
            7: {
                "instruction": "Got it. One moment while I pull available times for that day.",
                "navigation": {
                    "select_time": 8,
                },
            },
            8: {
                "instruction": "Perfect! I have booked your appointment for [date and time]. Is there anything else I can assist you with?",
                "navigation": {
                    "no_further_questions": 9,
                },
            },
            9: {
                "instruction": "Thank you for calling Company X. Have a great day!",
                "navigation": {
                    "end_call": 10,
                    "still_need_help": 11,
                    "off_rails": 12,
                },
            },
            10: {
                "instruction": "Goodbye! Have a great day!",
                "navigation": "terminate",
            },
            11: {
                "instruction": "I can schedule that for you. What day are you looking for us to come out?",
                "navigation": {
                    "provide_day": 7,
                    "node_13": 13,
                },
            },
            12: {
                "instruction": "I would love to help answer that question, but I can only schedule appointments. Would you like to schedule an appointment?",
                "navigation": {
                    "book_appointment": 1,
                    "other_services": 5,
                    "end_call": 10,
                    "node_15": 15,
                },
            },
            13: {
                "instruction": "You mentioned needing specialized service. Could you clarify the type of service requested?",
                "navigation": {
                    "special_service_A": 14,
                    "go_back": 12,
                },
            },
            14: {
                "instruction": "Special Service A might require additional fees. Are you comfortable with that?",
                "navigation": {
                    "yes_extra_fees": 5,
                    "no_cancel": 10,
                    "node_16": 16,
                },
            },
            15: {
                "instruction": "I understand you have a question about your recent invoice. Would you like me to transfer you to billing?",
                "navigation": {
                    "transfer_billing": 5,
                    "schedule_appointment": 1,
                    "node_17": 17,
                },
            },
            16: {
                "instruction": "We’ll need a contract for that. Do you have an existing contract number?",
                "navigation": {
                    "has_contract": 18,
                    "no_contract": 19,
                },
            },
            17: {
                "instruction": "It seems your question is out of scope. I can provide a frequently asked questions document or schedule an appointment. Which would you prefer?",
                "navigation": {
                    "faqs": 20,
                    "schedule_again": 1,
                },
            },
            18: {
                "instruction": "Please provide the contract number, and I’ll check if it’s valid.",
                "navigation": {
                    "contract_provided": 21,
                    "go_back": 14,
                },
            },
            19: {
                "instruction": "Without a contract, I can’t proceed. Would you like to speak to a sales representative?",
                "navigation": {
                    "sales_rep": 22,
                    "end_call": 10,
                },
            },
            20: {
                "instruction": "Here is the FAQ link: companyx.com/faq. Does this answer your questions?",
                "navigation": {
                    "faq_helpful": 9,
                    "node_23": 23,
                },
            },
            21: {
                "instruction": "Great, I see your contract is active. Do you need to add any special notes to your account?",
                "navigation": {
                    "add_notes": 24,
                    "no_notes": 9,
                },
            },
            22: {
                "instruction": "Please hold while I attempt to transfer you to a sales representative.",
                "navigation": "terminate",
            },
            23: {
                "instruction": "I’m sorry the FAQ link wasn’t helpful. Let me route you to a more specialized team.",
                "navigation": {
                    "specialized_team": 5,
                    "node_25": 25,
                },
            },
            24: {
                "instruction": "All notes are updated. Anything else we can assist you with?",
                "navigation": {
                    "no_further_questions": 9,
                    "add_more_notes": 21,
                },
            },
            25: {
                "instruction": "You appear to have more detailed inquiries. Could you describe them in detail?",
                "navigation": {
                    "describe_inquiry": 26,
                    "end_call": 10,
                },
            },
            26: {
                "instruction": "Thank you. Based on your description, we might have to escalate this to Tier 2 support.",
                "navigation": {
                    "escalate_tier2": 27,
                    "book_appointment": 1,
                },
            },
            27: {
                "instruction": "Tier 2 support is available 9 AM - 6 PM on weekdays. Shall I connect you now?",
                "navigation": {
                    "connect_tier2": 5,
                    "no_connect": 28,
                },
            },
            28: {
                "instruction": "Would you prefer to schedule a callback from Tier 2 support?",
                "navigation": {
                    "schedule_callback": 1,
                    "decline_help": 10,
                    "node_29": 29,
                },
            },
            29: {
                "instruction": "I’m sensing this issue might actually require an in-person appointment. Please confirm?",
                "navigation": {
                    "in_person_yes": 6,
                    "in_person_no": 10,
                },
            },
            30: {
                "instruction": "This node might cause confusion regarding shipping or inventory. Are you checking on an order?",
                "navigation": {
                    "order_status": 31,
                    "no_order": 9,
                },
            },
            31: {
                "instruction": "We have no record of your order. Please confirm your order ID, or we can schedule an appointment to discuss further.",
                "navigation": {
                    "confirm_order": 32,
                    "no_order_id": 10,
                },
            },
            32: {
                "instruction": "Order ID found. Let me direct you to the logistics team.",
                "navigation": "terminate",
            },
        }

        self.node_embeddings = {}
        for node_id, node_data in self.full_map.items():
            text = node_data["instruction"]
            self.node_embeddings[node_id] = self.model.encode(text)

    def get_navigation_map(self, nodes=None):
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

    def get_submap_upto_node(self, target_node):
        """
        Build a 'path' from node 0 to the 'target_node' (if possible), then collect:
          - all nodes on that path
          - the children of each node on that path (if they exist)
        Returns a sub-map (dict) that you can pass to the LLM.
        """
        # 1. Find path from 0 to target_node (DFS or BFS). We'll do a simple DFS here.
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
            # If there's no path, just return a single-node map if it exists
            return (
                {target_node: self.full_map[target_node]}
                if target_node in self.full_map
                else {}
            )

        # 2. Build submap from the nodes in 'path' + the children of those nodes
        submap_nodes = set(path)

        # For each node in the path, add its children
        for node_id in path:
            navigation = self.full_map[node_id].get("navigation", {})
            if isinstance(navigation, dict):
                for child in navigation.values():
                    if isinstance(child, int):
                        submap_nodes.add(child)

        # 3. Construct dictionary
        filtered_map = {}
        for node_id in submap_nodes:
            filtered_map[node_id] = self.full_map[node_id]

        return filtered_map
