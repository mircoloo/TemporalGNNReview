from torch_geometric import nn
import torch
import torch.nn.functional as F
import torch.nn


class Attention(torch.nn.Module):

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        self.ae = torch.nn.Parameter(torch.FloatTensor(116,1,1))
        self.ab = torch.nn.Parameter(torch.FloatTensor(116,1,1))

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)


        mix = attention_weights*(context.permute(0,2,1))

        delta_t = torch.flip(torch.arange(0, query_len), [0]).type(torch.float32).to('cuda')
        delta_t = delta_t.repeat(116,1).reshape(116,1,query_len)
        bt = torch.exp(-1*self.ab * delta_t)
        term_2 = F.relu(self.ae * mix * bt)
        mix = torch.sum(term_2+mix, -1).unsqueeze(1)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class gru(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = torch.nn.GRU(input_size = input_size, hidden_size=hidden_size, batch_first=True)
    def forward(self, inputs):
        full, last  = self.gru1(inputs)
        return full,last


class HGAT(torch.nn.Module):
    def __init__(self, n_features, n_nodes, sequence_length, hidden_dim):
        super(HGAT, self).__init__()
        #self.tickers = tickers # removed from original tickers
        self.grup = gru(n_features,hidden_dim)  #or lstm
        self.attention = Attention(hidden_dim)
        self.hidden_dim = hidden_dim
        self.hatt1: nn.HypergraphConv = nn.HypergraphConv(hidden_dim, hidden_dim, use_attention=False, heads=4, concat=False, negative_slope=0.2, dropout=0.5, bias=False)
        self.hatt2 = nn.HypergraphConv(hidden_dim, hidden_dim, use_attention=False, heads=1, concat=False, negative_slope=0.2, dropout=0.5, bias=True)
        self.liear = torch.nn.Linear(hidden_dim,1)
    def forward(self,price_input,e):
        context,query  = self.grup(price_input)
        query = query.reshape(n_nodes,1,self.hidden_dim)
        output, weights = self.attention(query, context)
        output = output.reshape((n_nodes,self.hidden_dim))
        hatt1_output = self.hatt1(output,e)
        x = F.leaky_relu(hatt1_output, 0.2)
        hatt2_output = self.hatt2(x,e)
        x = F.leaky_relu(hatt2_output, 0.2)
        return F.leaky_relu(self.liear(x))
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = '/home/mbisoffi/tests/TemporalGNNReview/code/data/datasets/graph/SSE_Validation_2017-07-01_2018-07-01_14/graph_0.pt' 
sample = torch.load(data_path, weights_only=False)

x = sample.x # get the x
n_nodes, n_features, n_timestamps = x.shape[0], 5, int(x.shape[1]/5)
x = x.reshape(n_nodes, n_features, n_timestamps).permute(0,2,1).to(device)

# get the index attributes
edge_index = sample.edge_index.to(device)

#get the targets
target = sample.y.reshape(-1,1).to(device)





model = HGAT(n_nodes=n_nodes, n_features=n_features, sequence_length=n_timestamps, hidden_dim=1024).to(device)
print(f"Number of parameters={sum([p.numel() for p in model.parameters()])}")
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
print(model.parameters)

for epoch in range(100000):
    model.train()
    optimizer.zero_grad()
    output = model(x,edge_index)
    loss = criterion(output, target) 
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        model.eval()
        print(output.shape, target.shape)
        print(loss.item())
        trends = (torch.sigmoid(output) > 0.5).int()
        acc = sum( trends == target ) / (len(output))
        print(torch.sigmoid(output).squeeze(), trends.squeeze(), target.squeeze())
        print(f"acc={acc.item()}")
        break