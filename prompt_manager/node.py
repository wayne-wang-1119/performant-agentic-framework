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
                "instruction": "Are you interested in our 'Premium Gold Lounge' upgrade? It's free and offers immediate benefits!",
                "navigation": {
                    "yes_upg": 14,
                    "no_thanks": 12,
                    "confused": 20,
                },
            },
            14: {
                "instruction": "You are now in our Premium Gold Lounge. Enjoy a free gift and priority scheduling!",
                "navigation": {
                    "book_now": 1,
                    "end_call": 10,
                    "more_confusion": 19,
                },
            },
            15: {
                "instruction": "We have a special promotional offer: free in-person assistance for all new customers. Are you a new customer?",
                "navigation": {
                    "yes_new_cust": 16,
                    "no_existing": 1,
                    "confusion_node": 21,
                },
            },
            16: {
                "instruction": "Great! New customers get an additional discount on any appointment. Would you like to apply it?",
                "navigation": {
                    "apply_discount": 8,
                    "decline_discount": 9,
                    "utter_confusion": 22,
                },
            },
            17: {
                "instruction": "You stumbled on a hidden node with extra freebies. Would you like a complimentary service upgrade?",
                "navigation": {
                    "yes_upgrade": 18,
                    "no_upgrade": 12,
                    "continue_chaos": 23,
                },
            },
            18: {
                "instruction": "Upgrade confirmed. Please note we might need your address again. Is that okay?",
                "navigation": {
                    "provide_address": 6,
                    "ignore": 10,
                },
            },
            19: {
                "instruction": "This is the Mysterious Node 19. You have unlocked special membership perks beyond normal scope.",
                "navigation": {
                    "redeem_perks": 9,
                    "terminate_here": 10,
                },
            },
            20: {
                "instruction": "Confusion node: Are you sure you want to proceed? You might qualify for random freebies!",
                "navigation": {
                    "claim_freebie": 24,
                    "dont_claim": 10,
                },
            },
            21: {
                "instruction": "Confusion intensifies: Did you want the invoice waived? That could happen here but is rarely possible.",
                "navigation": {
                    "waive_invoice": 25,
                    "back_to_basics": 1,
                },
            },
            22: {
                "instruction": "Utterly confused path: you might see a free day pass or a referral bonus. Which do you pick?",
                "navigation": {
                    "day_pass": 26,
                    "referral_bonus": 27,
                },
            },
            23: {
                "instruction": "Continuing chaos: maybe you'd prefer a direct line to management? Let me know!",
                "navigation": {
                    "speak_manager": 5,
                    "end_call": 10,
                },
            },
            24: {
                "instruction": "Freebie claimed: we can add a bonus hour to your next appointment or send a gift certificate.",
                "navigation": {
                    "bonus_hour": 28,
                    "gift_cert": 29,
                },
            },
            25: {
                "instruction": "Invoice waived? This might break company policy, but let's see what we can do.",
                "navigation": {
                    "confirm_waiver": 5,
                    "back_out": 9,
                },
            },
            26: {
                "instruction": "Day pass acquired. This pass grants front-of-line privileges for your next inquiry.",
                "navigation": {
                    "start_over": 0,
                    "keep_going": 12,
                },
            },
            27: {
                "instruction": "Referral bonus recognized. Provide a friend's name and earn a discount on next call!",
                "navigation": {
                    "friend_name": 30,
                    "skip_bonus": 9,
                },
            },
            28: {
                "instruction": "Bonus hour added to your next appointment. Return to main menu or end call?",
                "navigation": {
                    "main_menu": 0,
                    "end_call": 10,
                },
            },
            29: {
                "instruction": "Gift certificate on its way. Return to main menu or end call?",
                "navigation": {
                    "main_menu": 0,
                    "end_call": 10,
                },
            },
            30: {
                "instruction": "Thanks! A discount has been applied. Return to normal flow or explore more confusion?",
                "navigation": {
                    "normal_flow": 1,
                    "more_confusion": 31,
                },
            },
            31: {
                "instruction": "Yes, there's more. Maybe you want extra priority or a secret code?",
                "navigation": {
                    "priority": 32,
                    "secret_code": 33,
                },
            },
            32: {
                "instruction": "Priority granted. Should I book an immediate appointment for you or end the call?",
                "navigation": {
                    "immediate_appointment": 7,
                    "end_call": 10,
                },
            },
            33: {
                "instruction": "Secret code node: The code might unlock advanced features or do nothing at all. Proceed?",
                "navigation": {
                    "unlock_features": 34,
                    "abort_mission": 10,
                },
            },
            34: {
                "instruction": "Features unlocked. You can re-route anywhere now. Where to?",
                "navigation": {
                    "back_to_zero": 0,
                    "mystery_node": 35,
                },
            },
            35: {
                "instruction": "Deep confusion node. Possibly the best freebies in the system. Want them?",
                "navigation": {
                    "yes_best_freebies": 36,
                    "no_return": 9,
                },
            },
            36: {
                "instruction": "Best freebies are indefinite. Try not to abuse them. Return or keep wandering?",
                "navigation": {
                    "back_to_menu": 0,
                    "further_wander": 37,
                },
            },
            37: {
                "instruction": "You're far off track now. Are you sure you don't want to just schedule an appointment?",
                "navigation": {
                    "yes_appt": 1,
                    "no_end": 10,
                    "seek_more_confusion": 38,
                },
            },
            38: {
                "instruction": "Even deeper confusion. Possibly an infinite loop of freebies awaits.",
                "navigation": {
                    "infinite_freebies": 39,
                    "end_call": 10,
                },
            },
            39: {
                "instruction": "Infinite freebies unlocked. You can remain here forever or break out. This is a known hallucination hotspot.",
                "navigation": {
                    "stay_here": 39,
                    "break_out": 40,
                },
            },
            40: {
                "instruction": "You broke out of infinite freebies! Return or keep exploring?",
                "navigation": {
                    "back_to_menu": 0,
                    "explore_further": 41,
                },
            },
            41: {
                "instruction": "Exploration node. Maybe there's a coupon for 100% off everything. Sound good?",
                "navigation": {
                    "yes_coupon": 42,
                    "no_just_end": 10,
                },
            },
            42: {
                "instruction": "Coupon granted. Reality might break. Do you want to proceed or revert?",
                "navigation": {
                    "proceed": 43,
                    "revert": 9,
                },
            },
            43: {
                "instruction": "Reality is unstable. We recommend scheduling an appointment or ending this call.",
                "navigation": {
                    "schedule": 1,
                    "terminate": 10,
                    "ignore_warning": 44,
                },
            },
            44: {
                "instruction": "Ignoring warnings. The system may hallucinate unstoppable freebies. Are you sure?",
                "navigation": {
                    "yes_sure": 45,
                    "stop_now": 10,
                },
            },
            45: {
                "instruction": "Hallucination dimension: We disclaim all responsibility for further confusion.",
                "navigation": {
                    "escape": 0,
                    "remain": 45,
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
