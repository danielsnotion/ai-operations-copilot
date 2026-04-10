class ConversationMemory:
    def __init__(self, max_turns=5):
        self.history = []
        self.max_turns = max_turns

    def add(self, user, agent):
        self.history.append({"user": user, "agent": agent})

        # keep only last N turns
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self):
        context = ""
        for turn in self.history:
            context += f"User: {turn['user']}\nAgent: {turn['agent']}\n"
        return context

    def clear(self):
        self.history = []