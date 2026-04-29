import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 32
EPOCHS = 5
LR = 1e-3
THRESH = 0.55
TP = 0.0010
SL = 0.0010
SPREAD = 0.0001

# ============================================
# SYMBOLIC TOKENS
# ============================================

TOKENS = ['B', 'I', 'U', 'D', 'X', 'W', 'w']
stoi = {s:i for i,s in enumerate(TOKENS)}
itos = {i:s for s,i in stoi.items()}
VOCAB_SIZE = len(TOKENS)


# ============================================
# ENCODER
# ============================================

def encode_symbolic(df: pd.DataFrame):
    seq = []

    for i in range(len(df)):
        c = df.iloc[i]

        body = abs(c['close'] - c['open'])
        rng = max(c['high'] - c['low'], 1e-9)
        ratio = body / rng

        u_wick = c['high'] - max(c['open'], c['close'])
        l_wick = min(c['open'], c['close']) - c['low']

        if ratio < 0.1:
            token = 'X'
        elif c['close'] > c['open']:
            token = 'B' if ratio > 0.6 else 'U'
        else:
            token = 'I' if ratio > 0.6 else 'D'

        if u_wick > rng * 0.6:
            token = 'W'
        if l_wick > rng * 0.6:
            token = 'w'

        seq.append(stoi[token])

    return np.array(seq)


# ============================================
# DATASET
# ============================================

def build_dataset(seq):
    X, y = [], []

    for i in range(len(seq) - SEQ_LEN - 1):
        X.append(seq[i:i+SEQ_LEN])
        y.append(seq[i+SEQ_LEN])

    return np.array(X), np.array(y)


# ============================================
# TRANSFORMER MODEL
# ============================================

class TransformerCLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, SEQ_LEN, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)


# ============================================
# TRAIN
# ============================================

def train_model(X, y):
    model = TransformerCLM(VOCAB_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X).long().to(DEVICE)
    y_t = torch.tensor(y).long().to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()

        logits = model(X_t)
        loss = loss_fn(logits, y_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_t).float().mean().item()

        print(f"Epoch {epoch}: loss={loss.item():.4f} acc={acc:.4f}")

    return model


# ============================================
# DECISION LOGIC
# ============================================

def token_to_action(token_id):
    t = itos[token_id]

    if t in ['B', 'U']:
        return "BUY"
    elif t in ['I', 'D']:
        return "SELL"
    else:
        return None


# ============================================
# BACKTEST
# ============================================

def backtest(model, X, df):
    model.eval()
    balance = 0
    trades = []

    X_t = torch.tensor(X).long().to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(X_t), dim=1).cpu().numpy()

    for i in range(len(probs)):
        pred = np.argmax(probs[i])
        action = token_to_action(pred)

        if action is None:
            continue

        price = df['close'].iloc[i + SEQ_LEN]

        entry = price + SPREAD if action == "BUY" else price - SPREAD

        future_prices = df['close'].iloc[i+SEQ_LEN+1:i+SEQ_LEN+10]

        hit_tp, hit_sl = False, False

        for fp in future_prices:
            if action == "BUY":
                if fp >= entry + TP:
                    hit_tp = True
                    break
                if fp <= entry - SL:
                    hit_sl = True
                    break
            else:
                if fp <= entry - TP:
                    hit_tp = True
                    break
                if fp >= entry + SL:
                    hit_sl = True
                    break

        pnl = TP if hit_tp else (-SL if hit_sl else 0)
        balance += pnl
        trades.append(pnl)

    winrate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0

    print(f"\nTrades: {len(trades)}")
    print(f"Winrate: {winrate:.3f}")
    print(f"PnL: {balance:.5f}")


# ============================================
# MAIN
# ============================================

def run(csv_path):
    df = pd.read_csv(csv_path)

    seq = encode_symbolic(df)
    X, y = build_dataset(seq)

    split = int(len(X) * 0.8)

    X_train, y_train = X[:split], y[:split]
    X_test = X[split:]

    print("Training Transformer CLM...")
    model = train_model(X_train, y_train)

    print("Backtesting...")
    backtest(model, X_test, df.iloc[split:])


if __name__ == "__main__":
    run("eurusd_m1.csv")
