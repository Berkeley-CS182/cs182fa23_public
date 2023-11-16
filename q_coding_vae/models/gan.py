import torch
import torch.nn as nn
from models import nns
from torch.nn import functional as F



class GAN(nn.Module):
    def __init__(self, name='gan', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.g = nns.Generator(z_dim=z_dim)
        self.d = nns.Discriminator()   # note that discriminator outputs logits

    def loss_nonsaturating(self, x_real, *, device):
        '''
        Arguments:
        
        - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
        - device (torch.device): 'cpu' by default

        Returns:
        - d_loss (torch.Tensor): nonsaturating discriminator loss
        - g_loss (torch.Tensor): nonsaturating generator loss
        '''
        batch_size = x_real.shape[0]
        z = torch.randn(batch_size, self.g.z_dim, device=device)


        ############################# TODO #############################

        # YOUR CODE STARTS HERE
        # Note : You might want to detach the generator output from the computation graph
        #        when training the discriminator (to avoid backpropagating through the generator)
        # You may find some or all of the below useful:
        #   - F.binary_cross_entropy_with_logits
        #   - F.logsigmoid
        raise NotImplementedError
        # YOUR CODE ENDS HERE

        return d_loss, g_loss