"""
This file implements a few data sets to help retrieve the pre-processed
data for inverse RNA training and testing.

Implemented data set list:
    MoveSets: < More details to be added >
    ...
"""

import numpy as np
import os
import pickle
import torch

from torch.utils import data
from .. import registry


@registry.DATA_SETS.register("move_sets")
class MoveSets(data.Dataset):
    pass


@registry.DATA_SETS.register("eterna_complete_ss")
class EternaCompleteSS(data.Dataset):
    """
    Load and pre-process the eterna data set for training.
    """
    def __init__(
            self,
            data_path: str):
        """
        Initialize data set with config settings.
        The data file should contain puzzle information in lines ordered as:
            - Puzzle name
            - Dot bracket of target
            - Solution
            - Locked bases
        If solution / locked bases are not provided, it would be auto-generated with
        all 'A' and no locks.

        :param data_path: path to the eterna data set txt file.
        """
        # Load the eterna data set.
        if not os.path.exists(data_path):
            raise ValueError('Cannot find eterna data file.')
        with open(data_path, 'r') as f:
            raw_data = f.read().splitlines()
            self.data = []
            for row in raw_data:
                puzzle_info = [x.strip() for x in row.split("    ")]
                if len(puzzle_info) == 2:
                    puzzle_info += ['A' * len(puzzle_info[-1]), 'o' * len(puzzle_info[-1])]
                    self.data.append(puzzle_info)

