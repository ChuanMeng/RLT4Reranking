import torch

class Expert(torch.nn.Module):
    def __init__(self, args):
        super(Expert, self).__init__()
        self.args=args

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model = args.d_transformer, nhead = args.nhead, dropout = args.dropout, batch_first = True)
        self.attention_layer = torch.nn.TransformerEncoder(encoder_layer, num_layers=args.num_transformer_layers)

    def forward(self, x):
        # [B, S, 256]
        out = self.attention_layer(x)
        # [B, S, 256]
        return out
    

class TowerCut(torch.nn.Module):
    def __init__(self, args):
        super(TowerCut, self).__init__()
        self.args = args
        self.cut_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.d_transformer, out_features=1),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.cut_layer(x)
        return out


class TowerClass(torch.nn.Module):
    def __init__(self, args):
        super(TowerClass, self).__init__()
        self.args = args
        self.classification_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.d_transformer, out_features=1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.classification_layer(x)
        return out


class TowerRerank(torch.nn.Module):
    def __init__(self, args):
        super(TowerRerank, self).__init__()
        self.args = args
        self.rerank_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.d_transformer, out_features=1),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x [B, S, 256]
        out = self.rerank_layer(x)
        # out [B, S, 1]
        return out


class MMOECut(torch.nn.Module):
    def __init__(self, args):
        super(MMOECut, self).__init__()

        # params
        self.args=args
         # the first embedding layer
        self.pre_encoding = torch.nn.LSTM(input_size = args.feature_dim, hidden_size = args.d_lstm, num_layers = args.num_lstm_layers,
        batch_first = True, bidirectional = True, dropout = args.dropout)

        # row by row
        self.softmax = torch.nn.Softmax(dim=1)
        # model
        self.experts = torch.nn.ModuleList([Expert(args) for _ in range(self.args.num_experts)])

        # [[256*S, 3], [256*S, 3], [256*S, 3]]
        self.w_gates = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(args.d_lstm * 2 * args.seq_len , self.args.num_experts), requires_grad=True) for _ in range(int(args.num_tasks))])

        if args.num_tasks == 3:
            self.towers = torch.nn.ModuleList([
                TowerClass(args), # relevance judgement
                TowerRerank(args), # re-ranking
                TowerCut(args) # the last one is RLT
            ])
        elif args.num_tasks == 2.1:
            self.towers = torch.nn.ModuleList([
                TowerClass(self.expert_hidden),
                TowerCut(self.expert_hidden)
            ])
        elif args.num_tasks == 2.2:
            self.towers = torch.nn.ModuleList([
                TowerRerank(self.expert_hidden),
                TowerCut(self.expert_hidden)
            ])

    def forward(self, data):
        # [B, 1000, 3]
        # 3: score, tf-idf representation, doc2vec embedding

        x = data["feature"]
        # get the experts output
        experts_in = self.pre_encoding(x)[0] # [B, S, 256]
        experts_o = [e(experts_in) for e in self.experts] # [[B, S, 256], [B, S, 256], [B, S, 256]]
        experts_o_tensor = torch.stack(experts_o) # [3, B, S, 256]

        # get the gates output
        batch_size = experts_in.shape[0] # B

        # experts_in.reshape(batch_size, -1) --> [B, S*256]
        # [B, S*256] @ [256*S, 3]--> [B, 3]
        gates_o = [self.softmax(experts_in.reshape(batch_size, -1) @ g) for g in self.w_gates] # [[B,3], [B,3], [B,3]]

        # multiply the output of the experts with the corresponding gates output
        # res = gates_o[0].t().unsqueeze(2).expand(-1, -1, self.experts_out) * expers_o_tensor
        # https://discuss.pytorch.org/t/element-wise-multiplication-of-the-last-dimension/79534
        # [3, B]
        # [3, B, 1]
        # [3, B, S]
        # [3, B, S, 1]
        # [3, B, S, 256]
        # towers_input [[3, B, S, 256], [3, B, S, 256], [3, B, S, 256]]
        towers_input = [g.t().unsqueeze(2).expand(-1, -1, self.args.seq_len).unsqueeze(3).expand(-1, -1, -1, self.args.d_transformer) * experts_o_tensor for g in gates_o]

        # [[B, S, 256], [B, S, 256], [B, S, 256]]
        towers_input = [torch.sum(ti, dim=0) for ti in towers_input]

        # get the final output from the towers
        final_output = [t(ti) for t, ti in zip(self.towers, towers_input)] # [[B, S, 1], [B, S, 1], [B, S, 1]]

        # get the output of the towers, and stack them
        # final_output = torch.stack(final_output, dim=1)

        return final_output # [[B, S, 1], [B, S, 1], [B, S, 1]]
