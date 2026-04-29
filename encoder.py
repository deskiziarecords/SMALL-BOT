import numpy as np
import pandas as pd
from typing import List, Dict


class AlphaEncoder:
    """
    High-fidelity market state encoder

    Design goals:
    - Scale invariant (ATR-normalized)
    - Stable (no noisy transforms like FFT)
    - Learnable (continuous features)
    - Context-aware (structure + regime)
    - Compatible (symbol output retained)
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    # --------------------------------------------------
    # Core encoding
    # --------------------------------------------------

    def encode(self, df: pd.DataFrame) -> List[Dict]:
        df = df.copy()

        # -------------------------------
        # Volatility (ATR)
        # -------------------------------
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # -------------------------------
        # Base measures
        # -------------------------------
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['avg_body'] = df['body'].rolling(20).mean()

        encoded = []

        for i in range(len(df)):
            if i < self.lookback:
                continue

            row = df.iloc[i]

            atr = max(row['atr'], 1e-9)
            rng = max(row['range'], 1e-9)
            body = row['body']

            # ==================================================
            # MICROSTRUCTURE FEATURES
            # ==================================================

            # Direction (-1, 0, +1)
            direction = np.sign(row['close'] - row['open'])

            # Body strength (momentum proxy)
            body_strength = body / atr

            # Range expansion (volatility burst)
            range_expansion = rng / atr

            # Wick structure
            u_wick = row['high'] - max(row['open'], row['close'])
            l_wick = min(row['open'], row['close']) - row['low']

            u_wick_ratio = u_wick / atr
            l_wick_ratio = l_wick / atr

            # Wick imbalance (-1 to +1)
            wick_imbalance = (u_wick - l_wick) / rng

            # ==================================================
            # LOCAL DYNAMICS
            # ==================================================

            # Compression (FIXED: absolute vs ATR)
            price_changes = df['close'].iloc[i-10:i].diff().abs()
            compression = (price_changes < (0.2 * atr)).mean()

            # Expansion (engulfing proxy)
            avg_body = row['avg_body'] if row['avg_body'] > 0 else 1.0
            expansion = body / avg_body

            # Momentum (short-term drift)
            momentum = (row['close'] - df['close'].iloc[i-5]) / atr

            # ==================================================
            # STRUCTURAL CONTEXT
            # ==================================================

            # Trend (multi-bar drift)
            trend = (row['close'] - df['close'].iloc[i-20]) / atr

            # Relative position in recent range
            recent_high = df['high'].iloc[i-20:i].max()
            recent_low = df['low'].iloc[i-20:i].min()

            range_pos = (row['close'] - recent_low) / (
                (recent_high - recent_low) + 1e-9
            )

            # Volatility regime (expansion vs contraction)
            atr_mean = df['atr'].iloc[i-50:i].mean()
            vol_regime = atr / (atr_mean + 1e-9)

            # ==================================================
            # FINAL FEATURE VECTOR
            # ==================================================

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

            # ==================================================
            # SYMBOLIC LAYER (optional)
            # ==================================================

            ratio = body / rng

            if ratio < 0.1:
                symbol = 'X'
            elif direction > 0:
                symbol = 'B' if ratio > 0.6 else 'U'
            else:
                symbol = 'I' if ratio > 0.6 else 'D'

            if u_wick > rng * 0.6:
                symbol = 'W'
            if l_wick > rng * 0.6:
                symbol = 'w'

            encoded.append({
                "features": features,
                "symbol": symbol
            })

        return encoded
