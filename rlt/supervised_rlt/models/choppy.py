import torch

class Choppy(torch.nn.Module):
    def __init__(self, args):
        super(Choppy, self).__init__()
        self.args=args
        self.embedding = torch.nn.Embedding(args.seq_len, args.d_transformer-args.feature_dim) # [S, 128-feature_dim]
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args.d_transformer, nhead=args.nhead, dropout=args.dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=args.num_transformer_layers)
        self.decison_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.d_transformer, out_features=1), # [S, 1]
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, data):
        x = data["feature"]
        pe= self.embedding(data["pos"]) # [B, S, 128-feature_dim]
        x = torch.cat([x, pe], dim=2) # [B, S, d]
        x = self.transformer_encoder(x) # [B, S, d]
        x = self.decison_layer(x) # [B, S, 1]
        return x


        