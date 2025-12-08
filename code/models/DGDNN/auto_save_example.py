"""
Example: Enable automatic attention saving during DGDNN testing

This shows how to automatically save attention visualizations for every forward pass
during model testing/evaluation.
"""

import sys
sys.path.append('/home/mbisoffi/tests/TemporalGNNReview/code')

# When calling the test function, simply add save_attention=True:

# Example 1: In run.py or your training script
"""
runner.test(
    test_dataset, 
    window_size=14, 
    num_nodes=128, 
    batch_size=1,
    save_attention=True,                          # Enable attention saving
    attention_save_dir='dgdnn_attention_outputs', # Output directory
    max_saves=50                                  # Save first 50 forward passes
)
"""

# Example 2: Programmatic usage
"""
from model_runners.dgdnn_runner import DGDNNRunner
from models.DGDNN.dgdnn import DGDNN

# Create model and runner
model = DGDNN(...)
runner = DGDNNRunner(model, device='cuda', market_name='NASDAQ')

# Train model
runner.train(...)

# Test with attention saving enabled
results = runner.test(
    test_dataset=test_dataset,
    window_size=14,
    num_nodes=128,
    save_attention=True,           # Enable automatic saving
    attention_save_dir='attention_vis',
    max_saves=100                  # Limit to first 100 forward passes
)
"""

# The output will be organized as:
"""
attention_vis/
├── forward_pass_00000/
│   ├── theta_weights.png
│   ├── layer_0_attention.png
│   ├── layer_1_attention.png
│   └── summary.txt
├── forward_pass_00001/
│   ├── theta_weights.png
│   ├── layer_0_attention.png
│   ├── layer_1_attention.png
│   └── summary.txt
├── forward_pass_00002/
│   └── ...
...
"""

# Example 3: Enable/disable manually during custom loop
"""
model = DGDNN(...)

# Enable saving
model.enable_attention_saving(save_dir='my_attention_outputs', max_saves=20)

# Your custom forward passes
for data in test_loader:
    output = model(X, A)  # Attention is automatically saved
    # ... your code ...

# Disable saving
model.disable_attention_saving()
"""

print(__doc__)
