import torch.nn

def l1_regularization(model, l1_val):
    if l1_val <= 0:
        return 0 
    l1_criteria = nn.L1Loss(size_average=False)
    loss, regulariser_loss = 0, 0
    for parameter in model.parameters():
        regulariser_loss += l1_criteria(parameter, torch.zeros_like(parameter))
    loss += l1_val * regulariser_loss
    return loss