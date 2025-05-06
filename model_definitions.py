import torch.nn as nn

class BaseLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None):
        out, _ = self.lstm(x)        
        return self.fc(out[:, -1, :])

class LSTMwithEmbedding(BaseLSTM):
    def __init__(self, input_size=2, hidden_size=32, output_size=2, emb_dim=768):
        super().__init__(input_size, hidden_size, output_size)
        self.linear_h0 = nn.Linear(emb_dim, hidden_size)
    def forward(self, x, h0_vec):
        h0 = self.linear_h0(h0_vec).unsqueeze(0)            
        c0 = h0.clone().zero_()
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
