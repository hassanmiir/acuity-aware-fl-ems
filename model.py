import torch
import torch.nn as nn


class GRURiskPredictor(nn.Module):
    """
    GRU-based patient deterioration risk predictor.
    Returns raw logits — sigmoid applied externally
    (BCEWithLogitsLoss during training, torch.sigmoid during eval).
    """

    def __init__(self, input_size=7, hidden_size=128,
                 num_layers=2, output_size=1, dropout=0.3):
        super(GRURiskPredictor, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, output_size)

        # Xavier initialisation for stable training
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # x: (batch, window_size, input_size)
        out, _ = self.gru(x)
        out    = out[:, -1, :]        # last timestep
        out    = self.layer_norm(out) # LayerNorm more stable than BatchNorm for variable batch sizes
        out    = self.dropout(out)
        out    = self.classifier(out)
        return out.squeeze(-1)        # raw logits — NO sigmoid here