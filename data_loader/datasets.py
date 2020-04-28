"""
This file implements a few data sets to help retrieve the pre-processed
data for inverse RNA training and testing.

Implemented data set list:
    MoveSets: < More details to be added >
    ...
"""

import numpy as np
import os
import torch

import third_party.SentRNA.SentRNA.util.featurize_util as sent_rna_util

from torch.utils import data
import registry


@registry.DATA_SETS.register("move_sets")
class MoveSets(data.Dataset):
    pass


@registry.DATA_SETS.register("eterna")
class EternaDataSets(data.Dataset):
    """
    Load and pre-process the eterna data set for training.
    """
    def __init__(
            self,
            data_path: str,
            n_features: int,
    ):
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
        :param n_features: number of features to be used.
        """
        super(EternaDataSets, self).__init__()

        # Load the eterna data set.
        if not os.path.exists(data_path):
            raise ValueError('Cannot find eterna data file.')
        with open(data_path, 'r') as f:
            raw_data = f.read().splitlines()
            puzzle_data = []
            for row in raw_data:
                puzzle_info = [x.strip() for x in row.split("    ")]
                if len(puzzle_info) == 2:
                    puzzle_info += ['A' * len(puzzle_info[-1]), 'o' * len(puzzle_info[-1])]
                puzzle_data.append(puzzle_info)
            puzzle_data = np.array(puzzle_data)

        # pre-process using SentRNA feature extractions
        unique_puzzles, puzzle_count = np.unique(puzzle_data[:, 0], return_counts=True)
        puzzle_solution_count = {}
        for i, puzzle_name in enumerate(unique_puzzles):
            puzzle_solution_count[puzzle_name] = puzzle_count[i]
        features = sent_rna_util.compute_MI_features(
            puzzle_data, unique_puzzles, puzzle_solution_count, 50, 1, True, 0, 'rnaplot')
        np.random.shuffle(features)
        self.inputs, self.labels, self.rewards = sent_rna_util.parse_progression_dataset(
            puzzle_data, unique_puzzles, 1, features[:n_features], evaluate=False,
            shuffle=False, train_on_solved=True, MI_tolerance=1e-5, renderer='rnaplot')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return [self.inputs[index], self.labels[index], self.rewards[index]]


# This part is for testing if the data set is working as expected.
if __name__ == '__main__':
    eterna_data_path = os.path.join(
        os.path.dirname(__file__), '../third_party/SentRNA/data/test/eterna100.txt')
    print('Loading data from: ' + eterna_data_path)
    eterna_data_set = EternaDataSets(eterna_data_path, 20)
    print('Loaded eterna complete ss data with {} samples'.format(len(eterna_data_set)))
