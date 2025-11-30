import torch
def balanced_color_focal_loss(pred_ab, target_ab):
    # Define color regions in Lab space
    red_mask = (target_ab[:,0] > 45).float()    # A channel: red region
    green_mask = (target_ab[:,0] < -35).float() # A channel: green region
    yellow_mask = (target_ab[:,1] > 40).float() # B channel: yellow region
    blue_mask = (target_ab[:,1] < -40).float()  # B channel: blue region
    
    # Calculate errors with region masking
    red_error = torch.abs(pred_ab[:,0] - target_ab[:,0]) * red_mask
    green_error = torch.abs(pred_ab[:,0] - target_ab[:,0]) * green_mask
    yellow_error = torch.abs(pred_ab[:,1] - target_ab[:,1]) * yellow_mask
    blue_error = torch.abs(pred_ab[:,1] - target_ab[:,1]) * blue_mask
    
    # Apply balanced weights (maintain red/yellow focus while preserving greens/blues)
    return (
        3 * red_error**2 +    # Red emphasis
        1.5 * green_error**2 +  # Green preservation
        3 * yellow_error**2 + # Yellow emphasis
        1.5 * blue_error**2 +   # Blue preservation
        0.8 * torch.abs(pred_ab - target_ab).mean()  # Base color preservation
    ).mean()
def perceptual_loss(pred, target, feature_extractor):
    pred_features = feature_extractor(pred)
    target_features = feature_extractor(target)
    loss = 0.0
    for key in pred_features.keys():
        loss += nn.functional.mse_loss(pred_features[key], target_features[key])
    return loss

def color_focal_loss(pred_ab, target_ab):
    # Focus on red (A > 50) and yellow (B > 50)
    red_mask = (target_ab[:,0] > 50).float()
    yellow_mask = (target_ab[:,1] > 50).float()
    
    red_error = torch.abs(pred_ab[:,0] - target_ab[:,0]) * red_mask
    yellow_error = torch.abs(pred_ab[:,1] - target_ab[:,1]) * yellow_mask

    return (red_error**2 + yellow_error**2).mean() 
