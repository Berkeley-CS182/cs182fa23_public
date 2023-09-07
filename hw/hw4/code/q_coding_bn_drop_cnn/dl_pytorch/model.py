import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

        #######################################################################
        # TODO: Complete the implementation of the first convolutional block
        #
        # Feel free to check the hints of tensor shapes in the `forward` method
        #
        # Refer to pytorch.org for API documentations.
        # For NN modules: https://pytorch.org/docs/stable/nn.html
        #
        # The first conv block consists of a conv layer, an optional spatial
        #  batch normalization layer, a ReLU activation layer, and a maxpooling
        #  layer:
        # [conv1] -> ([bn1]) -> [relu1] -> [pool1]
        #
        # All conv layers in this neural network uses a kernel size of 3x3 and
        #  a padding of 1 on all sides (so that the height and width of the
        #  feature map does not change after the conv layer)
        #
        # Batch norm is enabled if and only if `do_batchnorm` is true. It
        #  should be a spatial batch norm layer for 2D images.
        #
        # All max-pooling layers in this neural network pools each non-
        #  overlapping 2x2 patch to a single pixel, shrinking the height/width
        #  of the feature map by 1/2.
        #
        # The first conv block has 16 filters
        #######################################################################
        self.conv1 = NotImplementedError()
        if do_batchnorm:
            self.bn1 = NotImplementedError()
        self.relu1 = NotImplementedError()
        self.pool1 = NotImplementedError()
        #######################################################################
        # End of your code
        #######################################################################

        #######################################################################
        # TODO: Implement the second convolutional block
        #
        # The second convolutional block has the same structure as the first,
        #  except that the conv layer has 32 filters
        #######################################################################
        raise NotImplementedError()
        #######################################################################
        # End of your code
        #######################################################################

        #######################################################################
        # TODO: Implement the third convolutional block
        #
        # The third convolutional block uses a strided conv layer with a stride
        #  of 2. It has 64 filters
        #
        # The conv layer is followed by an optional spatial batch norm layer,
        #  and a ReLU activation layer. No pooling in this block.
        #######################################################################
        raise NotImplementedError()
        #######################################################################
        # End of your code
        #######################################################################

        #######################################################################
        # TODO: Complete the 2-layer fully-connected classifier
        #
        # The input to this classifier is a flattened 1024-d vector for each
        #  input image.
        #
        # The classifier consists of two fully-connected layers, with a hidden
        #  dimension of 256 and an output dimension of 100. Dropout after the
        #  activation layer is enabled if and only if `p_dropout > 0.0`
        # [fc1] -> [relu4] -> ([drop]) -> [fc2]
        #
        # Feel free to check the hints of tensor shapes in the `forward` method
        #
        #######################################################################
        self.fc1 = NotImplementedError()
        self.relu4 = NotImplementedError()
        if p_dropout > 0.0:
            self.drop = NotImplementedError()
        self.fc2 = NotImplementedError()
        #######################################################################
        # End of your code
        #######################################################################

    def forward(self, x):
        # The shape of `x` is [bsz, 3, 32, 32]

        x = self.conv1(x)  # [bsz, 16, 32, 32]
        if self.do_batchnorm:
            x = self.bn1(x)
        x = self.pool1(self.relu1(x))  # [bsz, 16, 16, 16]

        x = self.conv2(x)  # [bsz, 32, 16, 16]
        if self.do_batchnorm:
            x = self.bn2(x)
        x = self.pool2(self.relu2(x))  # [bsz, 32, 8, 8]

        x = self.conv3(x)  # [bsz, 64, 4, 4]
        if self.do_batchnorm:
            x = self.bn3(x)
        x = self.relu3(x)

        x = torch.flatten(x, 1)  # [bsz, 1024]
        x = self.relu4(self.fc1(x))  # [bsz, 256]
        if self.p_dropout > 0.0:
            x = self.drop(x)
        x = self.fc2(x)  # [bsz, 100]
        return x
