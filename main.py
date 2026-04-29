import random
from model import load_model, predict
from risk_engine import RiskEngine
from trader import Trader
from logger import log_signal
from lambda7 import Lambda7

SEQ_LEN = 10
VOCAB = ['B','I','U','D','W','w','X']

def generate_fake_symbol():
    return random.choice(VOCAB)

def run():
    print("=== SMART-EXE CLEAN CORE ===")

    model = load_model("clm_eurusd.pt")
    risk = RiskEngine()
    trader = Trader()
    l7 = Lambda7()

    seq = []

    for step in range(50):
        sym = generate_fake_symbol()
        seq.append(sym)
        seq = seq[-SEQ_LEN:]

        if len(seq) < SEQ_LEN:
            continue

        probs = predict(model, seq)
        idx = probs.argmax()
        confidence = probs[idx]

        direction = "LONG" if idx in [0,2] else "SHORT"

        state = {
            "confidence": float(confidence),
            "direction": direction
        }

        decision = risk.evaluate(state)

        log_signal(state, decision)

        if decision["action"] == "ALLOW" and l7.validate(direction):
            trader.order(direction, decision["size"])

if __name__ == "__main__":
    run()
