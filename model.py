import torch
import torch.nn as nn

VOCAB = ['B','I','U','D','W','w','X']
MAP = {c:i for i,c in enumerate(VOCAB)}

class CandleLM(nn.Module):
    def __init__(self, dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(dim, len(VOCAB))

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)


def load_model(path):
    model = CandleLM()
    try:
        state = torch.load(path)
        model.load_state_dict(state)
        print(f"CLM loaded: {path}")
    except:
        print("⚠️ No model found, using random weights")
    model.eval()
    return model


def predict(model, seq):
    x = torch.tensor([[MAP[s] for s in seq]], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).numpy()[0]

    return probs
