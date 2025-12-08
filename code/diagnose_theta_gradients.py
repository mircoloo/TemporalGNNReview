"""
Script to diagnose why DGDNN theta parameters are not learning

This script checks:
1. If theta and T parameters have requires_grad=True
2. If gradients are flowing to theta during backpropagation
3. The effect of masking with adjacency matrix on gradient flow
"""

import torch
import torch.nn as nn
from models.DGDNN.dgdnn import DGDNN
from torch_geometric.utils import to_dense_adj

def check_parameter_gradients(model):
    """Check which parameters have gradients enabled"""
    print("="*70)
    print("CHECKING PARAMETER GRADIENT STATUS")
    print("="*70)
    
    all_learnable = True
    for name, param in model.named_parameters():
        if 'theta' in name or 'T' in name.split('.')[0]:
            status = "✓ Learnable" if param.requires_grad else "✗ FROZEN"
            if not param.requires_grad:
                all_learnable = False
            print(f"{name:40s} {status:15s} Shape: {tuple(param.shape)}")
    
    if all_learnable:
        print("\n✓ All theta and T parameters are learnable (requires_grad=True)")
    else:
        print("\n⚠️ WARNING: Some parameters are frozen!")
    
    return all_learnable


def test_gradient_flow(num_nodes=10, expansion_step=4, batch_size=2):
    """Test if gradients flow to theta during backpropagation"""
    print("\n" + "="*70)
    print("TESTING GRADIENT FLOW TO THETA")
    print("="*70)
    
    # Create a small DGDNN model
    config = {
        'diffusion_size': [5, 16, 16, 16],
        'embedding_size': [32, 32, 32],
        'embedding_hidden_size': 32,
        'embedding_output_size': 16,
        'raw_feature_size': 16,
        'classes': 1,
        'layers': 3,
        'num_nodes': num_nodes,
        'expansion_step': expansion_step,
        'num_heads': 2,
        'active': [True, True, True]
    }
    
    model = DGDNN(**config)
    model.train()
    
    # Create dummy data
    X = torch.randn(batch_size, num_nodes, 5, requires_grad=True)
    A = torch.rand(batch_size, num_nodes, num_nodes)
    A = (A + A.transpose(1, 2)) / 2  # Make symmetric
    A = (A > 0.5).float()  # Binary adjacency
    
    print(f"\nInput shapes:")
    print(f"  X: {X.shape}")
    print(f"  A: {A.shape}")
    print(f"  Adjacency density: {(A[0] > 0).sum().item() / (num_nodes * num_nodes) * 100:.1f}%")
    
    # Store initial theta values
    initial_theta = model.theta.data.clone()
    
    # Forward pass
    output = model(X, A)
    
    # Create a dummy loss
    target = torch.randint(0, 2, (batch_size, num_nodes, 1)).float()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check if theta has gradients
    print(f"\n🔍 Checking theta gradients after backward pass:")
    for l in range(config['layers']):
        theta_grad = model.theta.grad[l] if model.theta.grad is not None else None
        if theta_grad is not None:
            print(f"  Layer {l}:")
            print(f"    Gradient: {theta_grad.numpy()}")
            print(f"    Gradient norm: {theta_grad.norm().item():.6f}")
            print(f"    All zeros?: {(theta_grad.abs() < 1e-10).all().item()}")
            
            if (theta_grad.abs() < 1e-10).all():
                print(f"    ⚠️ WARNING: Gradients are zero! Theta won't update!")
            else:
                print(f"    ✓ Gradients are non-zero")
        else:
            print(f"  Layer {l}: ✗ NO GRADIENT!")
    
    # Simulate optimizer step
    with torch.no_grad():
        lr = 0.01
        if model.theta.grad is not None:
            model.theta -= lr * model.theta.grad
    
    # Check if theta changed
    print(f"\n📊 Checking if theta values changed:")
    theta_diff = (model.theta.data - initial_theta).abs()
    for l in range(config['layers']):
        print(f"  Layer {l}:")
        print(f"    Before: {initial_theta[l].numpy()}")
        print(f"    After:  {model.theta.data[l].numpy()}")
        print(f"    Change: {theta_diff[l].numpy()}")
        print(f"    Max change: {theta_diff[l].max().item():.8f}")
        
        if theta_diff[l].max().item() < 1e-8:
            print(f"    ⚠️ PROBLEM: Theta didn't change!")
        else:
            print(f"    ✓ Theta changed")


def test_adjacency_masking_effect():
    """Test how adjacency masking affects gradient flow"""
    print("\n" + "="*70)
    print("TESTING ADJACENCY MASKING EFFECT ON GRADIENTS")
    print("="*70)
    
    num_nodes = 5
    expansion_step = 4
    
    # Create dummy theta and T
    theta = nn.Parameter(torch.ones(expansion_step) / expansion_step)
    T_slices = nn.Parameter(torch.randn(expansion_step, num_nodes, num_nodes))
    
    # Test with different adjacency densities
    densities = [0.2, 0.5, 0.8, 1.0]
    
    for density in densities:
        print(f"\n--- Testing with adjacency density: {density*100:.0f}% ---")
        
        # Create adjacency matrix with specific density
        A = torch.rand(num_nodes, num_nodes)
        A = (A + A.T) / 2  # Symmetric
        A = (A < density).float()
        
        # Compute q with masking (current implementation)
        q = torch.einsum('k,kij->ij', theta, T_slices)
        q_masked = q * A  # ⚠️ This is the problematic line
        
        # Create dummy loss
        x = torch.randn(num_nodes, 10)
        q_sparse = q_masked.to_sparse()
        out = torch.sparse.mm(q_sparse, x)
        loss = out.sum()
        
        # Backward
        theta.grad = None
        T_slices.grad = None
        loss.backward()
        
        # Check gradients
        theta_grad_norm = theta.grad.norm().item() if theta.grad is not None else 0
        T_grad_norm = T_slices.grad.norm().item() if T_slices.grad is not None else 0
        
        print(f"  Adjacency edges: {(A > 0).sum().item()} / {num_nodes*num_nodes}")
        print(f"  Theta gradient norm: {theta_grad_norm:.6f}")
        print(f"  T gradient norm: {T_grad_norm:.6f}")
        
        # Check how many theta gradients are non-zero
        if theta.grad is not None:
            non_zero = (theta.grad.abs() > 1e-10).sum().item()
            print(f"  Non-zero theta gradients: {non_zero}/{expansion_step}")


def print_recommendations():
    """Print recommendations for fixing the theta learning problem"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS TO FIX THETA LEARNING")
    print("="*70)
    
    print("""
The problem is in models/DGDNN/ggd.py, line 26:

    q = q * a  # ⚠️ This masking can block gradients!

This multiplication zeros out many elements of q, which can prevent
gradients from flowing back to theta.

POTENTIAL SOLUTIONS:

1. **Remove hard masking** (if theoretically justified):
   Instead of: q = q * a
   Use: q = q  # No masking
   
   Rationale: Let the model learn the full diffusion matrix

2. **Use soft masking with adjacency as guidance**:
   Instead of: q = q * a
   Use: q = q * (1 + alpha * a)  # where alpha is a hyperparameter
   
   Rationale: Adjacency influences but doesn't completely block

3. **Add residual connection**:
   Instead of: q = q * a
   Use: q = (1 - beta) * q + beta * (q * a)
   
   Rationale: Blend learned and masked diffusion

4. **Use adjacency differently**:
   Instead of: q = q * a
   Use: q = torch.softmax(q + mask * a, dim=-1)
   
   Rationale: Adjacency as bias term in attention-like mechanism

5. **Add gradient scaling**:
   After: q = q * a
   Add: q = q + 0.01 * q.detach()  # Small gradient bypass
   
   Rationale: Ensure some gradient always flows

RECOMMENDED: Try solution #2 or #3 first, as they maintain some
relationship with the adjacency while allowing gradients to flow.

Also check:
- Learning rate for theta might need to be different (use parameter groups)
- Add L2 regularization to encourage theta diversity
- Use different initialization for theta (not uniform)
""")


if __name__ == "__main__":
    print("\n" + "🔍 DGDNN THETA GRADIENT DIAGNOSTIC TOOL")
    print("="*70)
    
    # Test 1: Check if parameters are learnable
    config = {
        'diffusion_size': [5, 16, 16, 16],
        'embedding_size': [32, 32, 32],
        'embedding_hidden_size': 32,
        'embedding_output_size': 16,
        'raw_feature_size': 16,
        'classes': 1,
        'layers': 3,
        'num_nodes': 10,
        'expansion_step': 4,
        'num_heads': 2,
        'active': [True, True, True]
    }
    model = DGDNN(**config)
    check_parameter_gradients(model)
    
    # Test 2: Check gradient flow
    test_gradient_flow()
    
    # Test 3: Check adjacency masking effect
    test_adjacency_masking_effect()
    
    # Print recommendations
    print_recommendations()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
