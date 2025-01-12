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
                "navigation": "terminate",
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
                "instruction": "We can re-schedule your appointment to next month or next year. Which do you prefer?",
                "navigation": {
                    "next_month": 7,  # Loops back to providing a day/time
                    "next_year": 14,
                    "back_to_main": 0,
                },
            },
            14: {
                "instruction": "Confirming next-year scheduling. You might lose your current slot. Continue?",
                "navigation": {
                    "confirm_loss": 8,  # Goes to booking final
                    "abort": 9,  # Standard exit
                    "further_confusion": 15,
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
                    "general_support": 5,  # Goes to terminate
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
