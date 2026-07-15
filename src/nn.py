# TODO upgrade to this after XGBoost
from torch import nn
import torch.nn.functional as F

class PitchSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_pitch_types):
        super().__init__()
        
        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # head 1: pitch type classification
        self.pitch_type_head = nn.Linear(hidden_dim, num_pitch_types)
        
        # head 2: location regression (plate_x, plate_z)
        self.location_head = nn.Linear(hidden_dim, 2)
    
    def forward(self, x, arsenal_mask=None):
        shared = self.trunk(x)
        
        # pitch type
        logits = self.pitch_type_head(shared)
        if arsenal_mask is not None:
            logits = logits.masked_fill(~arsenal_mask.bool(), float("-inf"))
        pitch_probs = F.softmax(logits, dim=-1)
        
        # location
        location = self.location_head(shared)  # raw (x, z) prediction
        
        return pitch_probs, location
    
def compute_loss(pitch_probs, location_pred, pitch_type_true, location_true,
                 alpha=1.0, beta=1.0):
    # classification loss for pitch type
    pitch_loss = F.cross_entropy(pitch_probs, pitch_type_true)
    
    # regression loss for location
    # mask out first pitches where prev location was sentinel
    valid = (location_true != -99).all(dim=1)
    loc_loss = F.mse_loss(
        location_pred[valid], 
        location_true[valid]
    )
    
    return alpha * pitch_loss + beta * loc_loss