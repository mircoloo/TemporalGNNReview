from argparse import Namespace

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models.base_models import NCModel
import torch
from torch_geometric.utils import to_dense_adj


#--lr 0.0004 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0 --use-att 1


import torch
import numpy as np
import argparse  # Used to easily create an 'args' object
torch.autograd.set_detect_anomaly(True)
# Assume the model classes (BaseModel, NCModel, etc.) are defined in the same file or imported
# from models.base_models import NCModel 

# --- 1. Define all your hyperparameters in an 'args' object ---
# This object simulates passing arguments from a command line.
args = argparse.Namespace()

# 1. Prepare args (you need to define this or use argparse)
args = Namespace(
    manifold='Hyperboloid', 
    c=1.0,
    cuda=0, 
    device='cuda',
    feat_dim=5, 
    n_nodes=1026, 
    model='HGCN',  # must exist in models/encoders.py
    n_classes=1,
    pos_weight=False,
    l=8,
    num_layers=2,
    a=2,
    task='nc',
    dataset='pubmed',
    act='relu',
    dim=16,
    dropout=0.4,
    bias=True,
    weight_decay=0.0001,    
    use_att=True,
)

args = Namespace(
        p='../data/2013-01-01',
        m='NASDAQ',
        t=None,
        l=20,
        u=256,
        s=10,
        r=0.001,
        a=10,
        gpu=0,
        emb_file='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy',
        rel_name='sector_industry',
        inner_prod=0,
        lr=0.001,
        dropout=0.2,
        cuda=0,
        epochs=5000,
        weight_decay=0.0001,
        optimizer='Adam',
        momentum=0.999,
        patience=100,
        seed=None,
        log_freq=5,
        eval_freq=1,
        save=0,
        save_dir=None,
        sweep_c=0,
        lr_reduce_freq=None,
        gamma=0.5,
        print_epoch=True,
        grad_clip=True,
        min_epochs=100,
        task='nc',
        model='HGCN',
        dim=5,
        manifold='PoincareBall',
        c=1.0,
        pretrained_embeddings=None,
        pos_weight=0,
        num_layers=10,
        bias=1,
        act='relu',
        n_heads=2,
        alpha=0.2,
        double_precision=0,
        use_att=0,
        dataset='pubmed',
        val_prop=0.05,
        test_prop=0.1,
        use_feats=1,
        normalize_feats=1,
        normalize_adj=1,
        split_seed=1234,
    ) 

p = '/home/mbisoffi/tests/TemporalGNNReview/code/data/datasets/graph/SSE_Test_2018-01-01_2019-12-31_14/graph_0.pt'
data = torch.load(p, weights_only=False)
num_nodes = data.x.shape[0]  # Number of nodes


args.feat_dim = 5  # Assuming each node has 5 features
args.n_nodes = num_nodes  # Number of nodes in the graph
args.l = int(data.x.shape[1] / 5)  # Length of the time series sequence for each node
args.n_classes = 1  # Assuming a binary classification task
# Training & Device Configuration
args.device =  'cuda' if torch.cuda.is_available() else 'cpu'
args.cuda = 0 if args.device == 'cuda' else -1


x = data['x'].to(args.device)  # Node features
x = x.reshape((args.n_nodes, args.feat_dim, args.l)).permute(0,2,1)  # Reshape to (n_nodes, l, feat_dim)
num_edges = data['edge_index'].shape[1]  # Number of edges
adj = to_dense_adj(edge_index=data.edge_index, edge_attr=data.edge_attr).squeeze().to(args.device)  # Adjacency matrix


print(x.shape, adj.shape)






print(f"Using device: {args.device}")

# --- 2. Instantiate the NCModel with your args ---
model = NCModel(args).to(args.device)
print(f"Number of model parameters: {sum([p.numel() for p in model.parameters()])}")
print("Model created successfully:")

def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

for submodule in model.modules():
        submodule.register_forward_hook(nan_hook)

# --- 3. Prepare your input data as PyTorch Tensors ---
# In a real scenario, you would load your data here.
# For this example, we'll create random dummy data with the correct shapes.

# Graph data
features = torch.rand(args.n_nodes, args.l, args.feat_dim).to(args.device) # Node features
adj = torch.rand(args.n_nodes, args.n_nodes).to(args.device)       # Adjacency matrix

# Financial data for the loss calculation
# base_price = torch.rand(args.n_nodes, 1).to(args.device)   # Base prices for each stock
# ground_truth = torch.rand(args.n_nodes, 1).to(args.device) # The true return ratios
# mask = torch.ones(args.n_nodes, 1).to(args.device)         # Mask to include all nodes

# Loss function hyperparameter
alpha = 0.5  # Weight for the ranking loss
# res = model.decode(embeddings, adj)
# while res.isnan().any():
#     print("Found NaN in the output, re-running decode...")
#     res = model.decode(embeddings, adj)
#     mask = res.isnan()
#     res[mask] = 0
# print(f"Resulting output shape: {res.shape} is nan in res:", res.isnan().any())
# print("Output of decode:", res, sum(mask))

targets =  torch.tensor(data.y, dtype=torch.float32).unsqueeze(1).to(args.device)  # Assuming binary classification targets
for epochs in range(1000):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    # Enable anomaly detection for more detailed debugging if NaNs persist
    # torch.autograd.set_detect_anomaly(True)
    
    emb = model.encode(x, adj)
    out = model.decode(emb, adj)
    
    # Disable anomaly detection if it was enabled
    # torch.autograd.set_detect_anomaly(False)
    
    print("Model output (logits) min:", out.min().item())
    print("Model output (logits) max:", out.max().item())
    if torch.isnan(out).any():
        print("NaNs found in model output!")
    if torch.isinf(out).any():
        print("Infs found in model output!")
    
    # Loss calculation
    loss = torch.nn.BCEWithLogitsLoss()(out, targets)
    loss.backward()
    
    # Optional: Gradient Clipping (Highly recommended for hyperbolic NNs)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Adjust max_norm as needed

    optimizer.step()
    
    # --- Multi-label Metric Calculation ---
    # Convert logits to probabilities using sigmoid
    probs = torch.sigmoid(out)
    
    # Threshold probabilities to get binary predictions (e.g., if probability > 0.5, predict 1, else 0)
    # Ensure predictions are on CPU and are of long type for sklearn functions
    pred = (probs > 0.5).cpu().long()
    
    # Ensure targets are also on CPU and of long type for sklearn functions
    targets_cpu = targets.cpu().long() 

    # 1. Element-wise Accuracy (what `accuracy_score` does by default for 2D inputs)
    element_wise_accuracy = accuracy_score(targets_cpu.reshape(-1), pred.reshape(-1)) # Flatten for sklearn to compare element-wise
    
    # 2. Sample-wise Accuracy (Exact Match Ratio)
    exact_match_accuracy = (pred == targets_cpu).all(dim=1).float().mean().item()
    
    # 3. Micro Precision, Recall, F1-Score
    micro_precision = precision_score(targets_cpu, pred, average='micro', zero_division=0)
    micro_recall = recall_score(targets_cpu, pred, average='micro', zero_division=0)
    micro_f1 = f1_score(targets_cpu, pred, average='micro', zero_division=0)
    
    # 4. Macro Precision, Recall, F1-Score
    macro_precision = precision_score(targets_cpu, pred, average='macro', zero_division=0)
    macro_recall = recall_score(targets_cpu, pred, average='macro', zero_division=0)
    macro_f1 = f1_score(targets_cpu, pred, average='macro', zero_division=0)

    print(f"Epoch {epochs+1}, Loss: {loss.item():.4f}")
    print(f"  Accuracy (Element-wise): {element_wise_accuracy:.4f}")
    print(f"  Accuracy (Exact Match): {exact_match_accuracy:.4f}")
    print(f"  Micro Metrics: Precision={micro_precision:.4f}, Recall={micro_recall:.4f}, F1={micro_f1:.4f}")
    print(f"  Macro Metrics: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}")

    print(f"{pred.squeeze()=} {targets_cpu.squeeze()=}")

# metrics = model.compute_metrics(
#     embeddings=embeddings, 
#     adj=adj, 
#     base_price=base_price, 
#     ground_truth=ground_truth, 
#     mask=mask, 
#     alpha=alpha, 
#     no_stocks=args.n_nodes
# )

# print("\nComputed metrics:")
# # The output is a dictionary containing your different loss components
# for key, value in metrics.items():
#     if isinstance(value, torch.Tensor):
#         print(f"- {key}: {value.item():.4f}")
#     else:
#         print(f"- {key}: {value}")


