import scipy.sparse
import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (nodes, features)
    def forward(self, seq, adj, sparse=True):
        seq_fts = self.fc(seq)
        if sparse:
                out = torch.spmm(adj, seq_fts)
        else:
                out = torch.bmm(adj, seq_fts)

        if self.bias is not None:
            out += self.bias
        return self.act(out)