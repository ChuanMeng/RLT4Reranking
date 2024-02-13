import torch


class BiCut(torch.nn.Module):
    def __init__(self, args):
        super(BiCut, self).__init__()
        self.args = args
        self.bilstm = torch.nn.LSTM(input_size=args.feature_dim, hidden_size=args.d_lstm,
                                    num_layers=args.num_lstm_layers, batch_first=True, bidirectional=True,
                                    dropout=args.dropout)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.d_lstm * 2, out_features=args.feedforward), # [256, 256]
            torch.nn.ReLU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(in_features=args.feedforward, out_features=2), # [256, 2]
        )

        self.softmax = torch.nn.Softmax(dim=2)
        

    def forward(self, data):
        x = data["feature"]
        x = self.bilstm(x)[0]
        x = self.ff(x) # [B,S,2]
        x = self.softmax(x) # [B, S, 2]
        
        return self.softmax(x)