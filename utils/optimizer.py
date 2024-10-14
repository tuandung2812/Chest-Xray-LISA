import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

# Define the WarmupDecayLR class
class WarmupDecayLR(LambdaLR):
    def __init__(self, optimizer, total_num_steps, warmup_min_lr, warmup_max_lr, warmup_num_steps, warmup_type="linear"):
        def lr_lambda(current_step):
            if current_step < warmup_num_steps:
                # Warmup phase
                if warmup_type == "linear":
                    return (warmup_max_lr - warmup_min_lr) / warmup_num_steps * current_step + warmup_min_lr / warmup_max_lr
                # Add other warmup types if needed
            else:
                # Decay phase
                progress = (current_step - warmup_num_steps) / (total_num_steps - warmup_num_steps)
                return max(0.0, 1.0 - progress)  # Linear decay after warmup
            
        super(WarmupDecayLR, self).__init__(optimizer, lr_lambda)