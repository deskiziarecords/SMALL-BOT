class RiskEngine:
    def evaluate(self, state):
        if state['confidence'] < 0.4:
            return {"action": "BLOCK", "reason": "low confidence"}

        return {
            "action": "ALLOW",
            "direction": state['direction'],
            "size": 0.01,
            "reason": "ok"
        }
