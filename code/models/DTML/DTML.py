import torch
import torch.nn as nn

class TimeAxisAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.tensor, rt_attn=False):
        # x: (D, W, L) # D: number of stocks / W: length of observations / L: number of features
        o, (h, _) = self.lstm(x) # o: (D, W, H) / h: (num_layers, D, H)
        
        # Take the last layer's hidden state
        h_last = h[-1]  # (D, H)
        
        # Compute attention scores
        score = torch.bmm(o, h_last.unsqueeze(-1))  # (D, W, H) x (D, H, 1) = (D, W, 1)
        tx_attn = torch.softmax(score.squeeze(-1), dim=1)  # (D, W)
        
        # Compute context vector
        context = torch.bmm(tx_attn.unsqueeze(1), o).squeeze(1)  # (D, 1, W) x (D, W, H) = (D, H)
        normed_context = self.lnorm(context)
        
        if rt_attn:
            return normed_context, tx_attn
        else:
            return normed_context, None
            
class DataAxisAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_rate=0.1):
        super().__init__()
        self.multi_attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.ReLU(),
            nn.Linear(4*hidden_size, hidden_size)
        )
        self.lnorm1 = nn.LayerNorm(hidden_size)
        self.lnorm2 = nn.LayerNorm(hidden_size)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, hm: torch.tensor, rt_attn=False):
        # Forward Multi-head Attention
        residual = hm
        # hm_hat: (D, H), dx_attn: (D, D) 
        hm_hat, dx_attn = self.multi_attn(hm, hm, hm)
        hm_hat = self.lnorm1(residual + self.drop_out(hm_hat))

        # Forward FFN
        residual = hm_hat
        # hp: (D, H)
        hp = torch.tanh(hm + hm_hat + self.mlp(hm + hm_hat))
        hp = self.lnorm2(residual + self.drop_out(hp))

        if rt_attn:
            return hp, dx_attn
        else:
            return hp, None

class DTML(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_heads, beta=0.1, drop_rate=0.1):
        super().__init__()
        self.beta = beta
        self.txattention = TimeAxisAttention(input_size, hidden_size, num_layers)
        self.dxattention = DataAxisAttention(hidden_size, n_heads, drop_rate)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, stocks, index, rt_attn=False):
        # stocks: (W, D, L) for a single time stamp
        # index: (W, 1, L) for a single time stamp
        # W: length of observations
        # D: number of stocks
        # L: number of features
        
        # Time-Axis Attention
        # c_stocks: (D, H) / tx_attn_stocks: (D, W)
        c_stocks, tx_attn_stocks = self.txattention(stocks.transpose(1, 0), rt_attn=rt_attn)
        # c_index: (1, H) / tx_attn_index: (1, W)
        c_index, tx_attn_index = self.txattention(index.transpose(1, 0), rt_attn=rt_attn)
        
        # Context Aggregation
        # Multi-level Context
        # hm: (D, H)
        hm = c_stocks + self.beta * c_index
        # The Effect of Global Contexts
        # effect: (D, D)
        effect = c_stocks.mm(c_stocks.transpose(0, 1)) + \
            self.beta * c_index.mm(c_stocks.transpose(1, 0)) + \
            self.beta**2 * torch.mm(c_index, c_index.transpose(0, 1)) 

        # Data-Axis Attention
        # hp: (D, H) / dx_attn: (D, D)
        hp, dx_attn_stocks = self.dxattention(hm, rt_attn=rt_attn)
        # output: (D, 1)
        output = self.linear(hp)

        return {
            'output': output,
            'tx_attn_stocks': tx_attn_stocks,
            'tx_attn_index': tx_attn_index,
            'dx_attn_stocks': dx_attn_stocks,
            'effect': effect
        }
    

if __name__ == "__main__":
    W = 14      # window length
    D = 712      # number of stocks
    L = 5      # features per timestep
    H = 100      # hidden size
    layers = 1
    heads = 5
    beta = 0.2
    # W: length of observations
    # D: number of stocks
    # L: number of features

    # Create random dummy inputs
    # stocks: (W, D, L)
    stocks = torch.randn(W, D, L)
    # index: (W, 1, L)
    index = torch.randn(W, 1, L)

    model = DTML(
        input_size=L,
        hidden_size=H,
        num_layers=layers,
        n_heads=heads,
        beta=beta,
        drop_rate=0.1
    )

    # Forward pass requesting attention maps
    out = model(stocks, index, rt_attn=True)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("Output score shape:", out["output"].shape)
    print("Time attention (stocks) shape:", out["tx_attn_stocks"].shape)
    print("Time attention (index) shape:", out["tx_attn_index"].shape)
    print("Data-axis attention shape:", out["dx_attn_stocks"].shape)
    print("Stock self effect shape:", out.get("stock_self_effect", None))
    print("Stock global effect shape:", out.get("stock_global_effect", None))

    # Quick sanity check of probability-like scores (if you wanted)
    probs = torch.sigmoid(out["output"])
    print("Sample probs:", probs[:5].squeeze(-1))