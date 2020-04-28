"""
This file implements the EternaBrain CNN based strategy for
inverse RNA puzzle solving.

The model learns from human player data and train CNN model to
predict the next move (e.g. location / value) of a human player.

More info about EternaBrain can be found at:
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007059
"""

import numpy as np
import torch
import torch.nn as nn
import RNA

from .. import registry


@registry.MODELS.register("EternaBrain")
class EternaBrain(nn.Module):
    """
    This class implements the strategy of EternaBrain.
    The class loads a pre-trained base and location network and predicts
    the potential RNA sequence given target secondary structure by updating
    an initial sequence iteratively.

    Note: this module should only be used for forward computation only now.
          And we don't ensure gradient back propagation works on this.
          To train the model, please train with sub-networks directly.
    """
    def __init__(self, **kwargs):
        super(EternaBrain, self).__init__()

        # Load base net.

        # Load position net.

    def forward(self, structure, **kwargs):
        pass


@registry.MODELS.register("EternaBrainBaseNet")
class EternaBrainBaseNet(nn.Module):
    """
    This class implements the EternaBrain base CNN structure
    to predict the base change.
    """
    def __init__(self, **kwargs):
        super(EternaBrainBaseNet, self).__init__()
        kernel_size = 9 if "kernel_size" not in kwargs else kwargs["kernel_size"]
        pad_size = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(1, 2, [kernel_size, kernel_size], padding=[pad_size, pad_size])

    def forward(self, structure, **kwargs):
        pass


@registry.MODELS.register("EternaBrainLocationNet")
class EternaBrainLocationNet(nn.Module):
    """
    This class implements the EternaBrain location CNN that
    predicts the location change of next move.
    """
    def __init__(self, **kwargs):
        super(EternaBrainLocationNet, self).__init__()

    def forward(self, structure, **kwargs):
        pass
