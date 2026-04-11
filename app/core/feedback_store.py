import json
import os
from datetime import datetime

FEEDBACK_FILE = "data/feedback.json"


class FeedbackStore:
    def __init__(self):
        if not os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "w") as f:
                json.dump([], f)

    def save_feedback(self, query, response, feedback):
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)

        data.append({
            "query": query,
            "response": response,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })

        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def get_negative_cases(self):
        with open(FEEDBACK_FILE, "r") as f:
            data = json.load(f)

        return [item for item in data if item["feedback"] == "negative"]