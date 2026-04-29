import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================
# CONFIG
# ============================================

LOOKBACK = 50
FEATURES = 12
EPOCHS = 5
LR = 1e-3
THRESH = 0.55
TP = 0.0010   # ~10 pips (EURUSD)
SL = 0.0010
SPREAD = 0.0001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# ENCODER (your final version simplified)
# ============================================

class AlphaEncoder:
    def __init__(self, lookback=50):
        self.lookback = lookback

    def encode(self, df: pd.DataFrame):
        df = df.copy()

        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['avg_body'] = df['body'].rolling(20).mean()

        X, y = [], []

        for i in range(len(df)):
            if i < self.lookback:
                continue

            row = df.iloc[i]
            atr = max(row['atr'], 1e-9)
            rng = max(row['range'], 1e-9)
            body = row['body']

            direction = np.sign(row['close'] - row['open'])
            body_strength = body / atr
            range_expansion = rng / atr

            u_wick = row['high'] - max(row['open'], row['close'])
            l_wick = min(row['open'], row['close']) - row['low']

            u_wick_ratio = u_wick / atr
            l_wick_ratio = l_wick / atr
            wick_imbalance = (u_wick - l_wick) / rng

            price_changes = df['close'].iloc[i-10:i].diff().abs()
            compression = (price_changes < (0.2 * atr)).mean()

            avg_body = row['avg_body'] if row['avg_body'] > 0 else 1.0
            expansion = body / avg_body

            momentum = (row['close'] - df['close'].iloc[i-5]) / atr
            trend = (row['close'] - df['close'].iloc[i-20]) / atr

            recent_high = df['high'].iloc[i-20:i].max()
            recent_low = df['low'].iloc[i-20:i].min()

            range_pos = (row['close'] - recent_low) / (
                (recent_high - recent_low) + 1e-9
            )

            atr_mean = df['atr'].iloc[i-50:i].mean()
            vol_regime = atr / (atr_mean + 1e-9)

            features = np.array([
                direction,
                body_strength,
                range_expansion,
                u_wick_ratio,
                l_wick_ratio,
                wick_imbalance,
                compression,
                expansion,
                momentum,
                trend,
                range_pos,
                vol_regime
            ], dtype=np.float32)

            # LABEL: next candle direction
            future = df['close'].iloc[i+1] - row['close'] if i+1 < len(df) else 0
            label = 1 if future > 0 else 0

            X.append(features)
            y.append(label)

        return np.array(X), np.array(y)


# ============================================
# MODEL (simple but effective)
# ============================================

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# ============================================
# TRAIN
# ============================================

def train_model(X, y):
    model = SimpleMLP().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.tensor(X).to(DEVICE)
    y_t = torch.tensor(y).long().to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_t).float().mean().item()

        print(f"Epoch {epoch}: loss={loss.item():.4f} acc={acc:.4f}")

    return model


# ============================================
# BACKTEST
# ============================================

def backtest(model, X, df):
    model.eval()

    balance = 0
    trades = []

    X_t = torch.tensor(X).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(X_t), dim=1).cpu().numpy()

    for i in range(len(probs)-1):
        p_down, p_up = probs[i]

        price = df['close'].iloc[i + LOOKBACK]

        action = None
        if p_up > THRESH:
            action = "BUY"
        elif p_down > THRESH:
            action = "SELL"

        if action is None:
            continue

        entry = price + SPREAD if action == "BUY" else price - SPREAD

        future_prices = df['close'].iloc[i+LOOKBACK+1:i+LOOKBACK+10]

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

        trades.append({
            "action": action,
            "p_up": float(p_up),
            "p_down": float(p_down),
            "pnl": pnl
        })

    wins = sum(1 for t in trades if t["pnl"] > 0)
    total = len(trades)
    winrate = wins / total if total > 0 else 0

    print(f"\nTrades: {total}")
    print(f"Winrate: {winrate:.3f}")
    print(f"PnL: {balance:.5f}")

    return trades


# ============================================
# MAIN
# ============================================

def run(csv_path):
    df = pd.read_csv(csv_path)

    encoder = AlphaEncoder()
    X, y = encoder.encode(df)

    print("Training...")
    model = train_model(X, y)

    print("Backtesting...")
    backtest(model, X, df)


if __name__ == "__main__":
    run("eurusd_m1.csv")
