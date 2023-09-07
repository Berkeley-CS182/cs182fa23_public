import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################################
# TODO: Design your own neural network
# You can define utility functions/classes here
#######################################################################
pass
#######################################################################
# End of your code
#######################################################################


class MyNeuralNetwork(nn.Module):
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

        #######################################################################
        # TODO: Design your own neural network
        #######################################################################
        raise NotImplementedError()
        #######################################################################
        # End of your code
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Design your own neural network
        #######################################################################
        raise NotImplementedError()
        #######################################################################
        # End of your code
        #######################################################################
