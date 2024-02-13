import torch

class AttnCut(torch.nn.Module):
    def __init__(self, args):
        super(AttnCut, self).__init__()
        self.args=args
        self.encoding_layer = torch.nn.LSTM(input_size = args.feature_dim, hidden_size = args.d_lstm, num_layers = args.num_lstm_layers,
                                            batch_first = True, bidirectional = True, dropout = args.dropout)

        transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args.d_transformer, nhead=args.nhead, dropout=args.dropout, batch_first=True)
        self.attention_layer = torch.nn.TransformerEncoder(transformer_encoder_layer, num_layers=args.num_transformer_layers)

        self.decison_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.d_transformer, out_features=1), # [256, 1]
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, data):
        x = data["feature"]
        x = self.encoding_layer(x)[0] # [B, S, 256]
        x = self.attention_layer(x) # [B, S, 256]
        x = self.decison_layer(x) #

        # print([x_.cpu().detach().numpy().tolist().index(max(x_.cpu().detach().numpy().tolist())) for x_ in x])
        return x

