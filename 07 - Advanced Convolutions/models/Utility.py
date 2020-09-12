import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from models.NetworkArchitecture import Net



def sgd_optimizer(model, learning_rate, momentum, l2_value = 0.0):
    """
    input parameters : 
    instance of model, desired learning rate, momentum value, L2 regularisation value
    
    returns Stocastic Gradient Descent optimizer.
    """
    return optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay = l2_value)


def steplr_scheduler(optimizer, step_size, gamma):
    """
    input parameters :
    model optimizer, lr changing step size, lr changing factor
    
    return step LR scheduler
    """

    return StepLR(optimizer, step_size = step_size, gamma = gamma)


def model_summary(model, input_size):
    """
    input parameters:
    dummy model, dimensions of input

    returns the model summary (parameters)
    """
    print(summary(model, input_size = input_size))



def set_seed(seed, cuda_present):
  """ This is done to reproduce the results """
  torch.manual_seed(seed)
  if cuda_present:
    torch.cuda.manual_seed(seed)


def initialize_cuda(seed):
    """ This is used to run on GPU """

    # Check CUDA availability
    cuda_present = torch.cuda.is_available()
    print('GPU Available?', cuda_present)

    # Initialize seed
    set_seed(seed, cuda_present)

    # Set device
    device = torch.device("cuda" if cuda_present else "cpu")

    return cuda_present, device

def cross_entropy_loss():
    return nn.CrossEntropyLoss()

def nll_loss():
  return nn.NLLLoss()

