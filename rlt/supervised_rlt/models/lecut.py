import torch

class LeCut(torch.nn.Module):
    def __init__(self, args):
        super(LeCut, self).__init__()
        self.args=args

        self.encoding_layer = torch.nn.LSTM(input_size=args.feature_dim, hidden_size=args.d_lstm , num_layers=args.num_lstm_layers, batch_first=True, bidirectional=True)

        self.embedding = torch.nn.Embedding(args.seq_len, args.d_position)  # [S, 32]

        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args.d_transformer, nhead=args.nhead, dropout=args.dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(transformer_encoder_layer, num_layers=args.num_transformer_layers)

        self.shorten_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.d_embd, out_features=1), # [768, 1]
            torch.nn.Softmax(dim=1)
        )

        self.decison_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.d_transformer, out_features=1), # [256, 1]
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, data):

        x = data["feature"]
        assert (self.args.feature_dim-1+self.args.d_embd)==x.size()[-1]

        x_feature = x[:,:,:(self.args.feature_dim-1)]
        # x[:, :, -768:] --> [B, S, 768]
        # self.shorten_layer(x[:, :, -768:]) --> [B, S, 1]
        x_embd = self.shorten_layer(x[:, :, -self.args.d_embd:])
        # [B, S, 9]
        x = torch.cat([x_feature, x_embd], 2)
        # [B, S, 224]
        x = self.encoding_layer(x)[0]
        pe = self.embedding(data["pos"]) # [B, S, 32]
        # [B, S, 256]
        x = torch.cat([x, pe], dim=2)
        x = self.transformer_encoder(x) # [B, S, 256]
        x = self.decison_layer(x) # [B, S, 1]

        return x
