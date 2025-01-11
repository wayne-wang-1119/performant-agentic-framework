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
                    "off rails": 12,
                },
            },
            10: {
                "instruction": "Goodbye! Have a great day!",
                "navigation": "terminate",
            },
            11: {
                "instruction": "I can schedule that for you what day are you looking for us to come out?",
                "navigation": {
                    "provide_day": 7,
                },
            },
            12: {
                "instruction": "I would love to help answer that question, but I am only able to schedule appointments. Would you like to schedule an appointment?",
                "navigation": {
                    "book_appointment": 1,
                    "other_services": 5,
                    "end_call": 10,
                },
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
