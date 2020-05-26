"""
This file implements the NEMO strategy for inverse RNA folding problem.
The original paper can be found at:
    https://www.biorxiv.org/content/10.1101/345587v1.full
"""
import RNA
import numpy as np
import os
import platform
import time
from typing import Sequence, Union
from functools import wraps

# Structure to hold timing information.
PERF_DATA = {}


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PERF_DATA:
            PERF_DATA[fn.__name__] = [0, []]
        PERF_DATA[fn.__name__][0] += 1
        PERF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling


def print_perf_data():
    for function_name, data in PERF_DATA.items():
        print("....Function {} called {} times in {} seconds. ".format(function_name, data[0], sum(data[1])))


def clear_perf_data():
    global PERF_DATA
    PERF_DATA = {}


def rna_inverse(
        target_fold: str,
        start_seq: str = None,
        max_iter: int = 2500,
        weight_prob: dict = None
):
    """
    Find a sequence that folds into the target secondary structure.

    :param target_fold: target fold secondary structure in dot bracket format.
    :param start_seq: the sequence to start the search.
    :param max_iter: max number of iteration to run.
    :param weight_prob: probabilities dictionary to be use for random sampling.
    :return: the sequence on success or None if max_iter is reached.
    """
    if not start_seq:
        start_seq = 'N' * len(target_fold)
    assert len(start_seq) == len(target_fold), "start_seq and target fold length mismatched"

    pair_table = RNA.ptable(target_fold)
    mismatch_table = _make_mismatch_table(pair_table)
    loop_table = _make_loop_table(pair_table)
    junction_table = _make_junction_table(pair_table, loop_table)
    strength_table = _make_strength_table(pair_table, mismatch_table)
    class_indices = _classify(pair_table, mismatch_table, junction_table)
    shuffle = np.arange(len(start_seq))
    stuck = 0

    current_sequence = list(start_seq)
    for iter_index in range(max_iter):
        filled_sequence = _fill_seq(
            target_fold, current_sequence, pair_table, mismatch_table, junction_table, weight_prob, class_indices)
        current_fold, _ = RNA.fold(''.join(filled_sequence))
        base_pair_distance = RNA.bp_distance(current_fold, target_fold)
        if base_pair_distance == 0:
            print('....solution found at iteration: {}'.format(iter_index))
            return ''.join(filled_sequence)

        # Build a new starting sequence based on the mismatch.
        current_pair_table = RNA.ptable(current_fold)
        retry = (np.array(pair_table) != np.array(current_pair_table))
        for j in range(1, len(pair_table)):
            retry[j] |= mismatch_table[j] > 0 and retry[mismatch_table[j]]
        current_mismatch_table = _make_mismatch_table(current_pair_table)
        current_loop_table = _make_loop_table(current_pair_table)

        for j in range(1, current_loop_table[0]):
            i = 1
            for i in range(1, len(current_pair_table)):
                if current_loop_table[i] == j:
                    break
            has_opened_pair = False
            closing_good = True
            k = i
            while True:
                while True:
                    k += 1
                    if k >= len(current_pair_table):
                        k = 1
                    if current_pair_table[k] == 0 and pair_table[k] > 0:
                        has_opened_pair = True
                    if current_pair_table[k] != 0:
                        break
                if current_pair_table[k] != pair_table[k]:
                    closing_good = False
                k = current_pair_table[k]
                if k == i or not closing_good:
                    break
            if closing_good and has_opened_pair:
                while True:
                    while True:
                        k += 1
                        if k >= len(current_pair_table):
                            k = 1
                        if current_pair_table[k] > 0 or current_mismatch_table[k] > 0:
                            retry[k] = True
                        if current_pair_table[k] != 0:
                            break
                    k = current_pair_table[k]
                    retry[k] = True
                    if k == i:
                        break

        # Set up new sequence for retry
        while True:
            for k in range(len(current_pair_table) - 1):
                r = np.random.randint(len(current_pair_table) - 2)
                if r >= k:
                    r += 1
                shuffle[k], shuffle[r] = shuffle[r], shuffle[k]

            c = 0
            for k in range(len(current_pair_table) - 1):
                i = shuffle[k]

                # skip if this is not a mis-fold spot
                if not retry[i + 1]:
                    continue

                # we don't want to reset and retry all mis-folded bases and pairs
                # so we use probabilities, first 1/1, then 1/2, 1/3 and so on
                # this guarantees that we will reset at least one base or pair,
                # and maybe a few more.
                c += 1
                if c > 0 and np.random.rand() > 1 / c:
                    continue

                if (filled_sequence[i] != 'N' and current_pair_table[i + 1] > 0
                        and filled_sequence[current_pair_table[i + 1] - 1] != 'N'):
                    if pair_table[i + 1] == 0 and pair_table[current_pair_table[i + 1] - 1]:
                        filled_sequence = _unset_bases(filled_sequence, pair_table, i)
                        filled_sequence = _unset_bases(filled_sequence, pair_table, current_pair_table[i + 1] - 1)
                    else:
                        if (start_seq[current_pair_table[i + 1] - 1] != 'N' or np.random.randint(
                                strength_table[i + 1] + strength_table[current_pair_table[i + 1]]) <
                                strength_table[i + 1]):
                            filled_sequence = _unset_bases(filled_sequence, pair_table, i)
                        else:
                            filled_sequence = _unset_bases(filled_sequence, pair_table, current_pair_table[i + 1] - 1)
                elif filled_sequence[i] != 'N':
                    filled_sequence = _unset_bases(filled_sequence, pair_table, i)

                if mismatch_table[i + 1] > 0:
                    filled_sequence = _unset_bases(filled_sequence, pair_table, mismatch_table[i + 1] - 1)

                if i > 0 and pair_table[i + 1] > 0 and pair_table[i] == 0 and mismatch_table[i] > 0:
                    j = i - 1
                    filled_sequence[j] = 'N'
                    filled_sequence = _unset_bases(filled_sequence, pair_table, mismatch_table[j + 1] - 1)
                    if current_pair_table[i + 1] == 0:
                        while True:
                            j -= 1
                            if j < 0:
                                break
                            filled_sequence = _unset_bases(filled_sequence, pair_table, j)
                            if pair_table[j + 1] > 0:
                                break
                if (pair_table[i + 1] > 1 and pair_table[pair_table[i + 1] - 1] == 0
                        and mismatch_table[pair_table[i + 1] - 1] > 0):
                    j = pair_table[i + 1] - 2
                    filled_sequence[j] = 'N'
                    filled_sequence = _unset_bases(filled_sequence, pair_table, mismatch_table[j + 1] - 1)

                if (0 < pair_table[i + 1] < len(pair_table) - 1 and pair_table[pair_table[i + 1] + 1] == 0
                        and mismatch_table[pair_table[i + 1] + 1] > 0):
                    j = pair_table[i + 1]
                    filled_sequence[j] = 'N'
                    filled_sequence = _unset_bases(filled_sequence, pair_table, mismatch_table[j + 1] - 1)

            # Make sure locked bases are untouched.
            for i, ch in enumerate(start_seq):
                if ch != 'N':
                    filled_sequence[i] = ch

            if not (set(filled_sequence) <= {'A', 'U', 'G', 'C'}):
                break
        if ''.join(current_sequence) == ''.join(filled_sequence):
            stuck += 1
            if stuck > 10:
                print('....Stuck, reset to start sequence')
                filled_sequence = list(start_seq)
        else:
            stuck = 0
        current_sequence = filled_sequence
        if (iter_index + 1) % 100 == 0:
            print('....finished iteration: {}'.format(iter_index + 1))
    return None


@profile
def compute_score(target_fold: str, sequence: Union[str, list]):
    """
    Compute score for certain RNA sequence given the target fold structure.

    :param target_fold: target fold secondary structure in dot bracket format.
    :param sequence: RNA sequence of same length as target fold.
    :return: tuple of score value and bool indicating if problem solved.
    """
    if isinstance(sequence, list):
        sequence = ''.join(sequence)
    current_fold, free_energy = RNA.fold(sequence)
    base_pair_distance = RNA.bp_distance(current_fold, target_fold)
    n_pairs = target_fold.count('(')
    score = 1.0 / (1.0 + base_pair_distance) if n_pairs == 0 else 1 - base_pair_distance / n_pairs / 2

    target_energy = RNA.energy_of_struct(sequence, target_fold)
    e_factor = 1.01 + target_energy - free_energy
    score *= e_factor if score < 0 else 1 / e_factor
    return score, base_pair_distance == 0


def _make_mismatch_table(pair_table: Sequence[int]):
    """
    Create a mismatch table based on the pair map.
    TODO: add more description about the mismatch table.
    TODO: this function is adapted from C++, to make it more pythonic later.

    :param pair_table: the pair table list of the paired index of each position.
    :return: a numpy array of index of positions.
    """
    mismatch = np.zeros(len(pair_table), dtype=np.int)
    for j in range(1, len(pair_table)):
        if pair_table[j] > 0:
            if (j < pair_table[0] and pair_table[j] > 1 and pair_table[j + 1] > 0 and
                    pair_table[pair_table[j] - 1] > 0 and pair_table[j + 1] != pair_table[j] - 1):
                mismatch[j + 1] = pair_table[j] - 1
                mismatch[pair_table[j] - 1] = j + 1
            continue
        if j < pair_table[0] and pair_table[j + 1] > 0 and pair_table[j + 1] + 1 <= pair_table[0]:
            mismatch[j] = pair_table[j + 1] + 1
            mismatch[pair_table[j + 1] + 1] = j
        elif j > 1 and pair_table[j - 1] and pair_table[j - 1] - 1 >= 1:
            mismatch[j] = pair_table[j - 1] - 1
            mismatch[pair_table[j - 1] - 1] = j
    return mismatch


def _make_loop_table(pair_table: Sequence[int]):
    """
    Create a loop table filled with loop index at each position.
    Loop index 0 indicates the position is not in a loop.
    Positions with same positive loop index are in the same loop.
    The first entry in the loop table indicates number of loops detected.
    TODO: this function is adapted from C++, to make it more pythonic later.

    :param pair_table: the pair table list of the paired index of each position.
    :return: a numpy array of index of positions.
    """
    loop = np.zeros(len(pair_table), dtype=np.int)
    loop_index = 1
    for j in range(1, len(pair_table)):
        if loop[j] == 0 and pair_table[j] > 1 and pair_table[pair_table[j] - 1] != j + 1:
            loop[j] = loop_index
            k = j
            while True:
                while True:
                    k += 1
                    if k > pair_table[0]:
                        k = 1
                    if pair_table[k] != 0:
                        break
                k = pair_table[k]
                loop[k] = loop_index
                if k == j:
                    break
            loop_index += 1
    loop[0] = loop_index
    return loop


def _make_junction_table(pair_table: Sequence[int], loop_table: Sequence[int] = None):
    """
    Create a junction table.

    :param pair_table: the pair table list of the paired index of each position.
    :param loop_table: the loop table created with _make_loop_table
    :return: a numpy array of loop index with loop element count larger than 2.
    """
    if loop_table is None:
        loop_table = _make_loop_table(pair_table)
    assert len(pair_table) == len(loop_table), "Pair table and loop table size mismatch"

    junction_table = np.array(loop_table)
    val, index, counts = np.unique(junction_table[1:], return_inverse=True, return_counts=True)
    val[counts <= 2] = 0
    junction_table[1:] = val[index]
    return junction_table


def _make_strength_table(pair_table: Sequence[int], mismatch_table: Sequence[int] = None):
    """
    Create a strength table.
    The "strength map" encodes a rough measure of the "fragility" of the
    structure at the considered index. Long helices are "strong", isolated
    base pairs, junctions, etc, are "weak".

    The map comes into play when deciding which side in a mis-folded pair
    should be mutated. The heuristic is that a "strong" domain should be
    able to take a mutation more easily than a weaker one.

    TODO: this function is adapted from C++, to make it more pythonic later.

    :param pair_table: the pair table list of the paired index of each position.
    :param mismatch_table: the mismatch table created with _make_mismatch_table
    :return: a numpy array of strength map.
    """
    if mismatch_table is None:
        mismatch_table = _make_mismatch_table(pair_table)
    assert len(pair_table) == len(mismatch_table), "Length of pair table and mismatch table is different"

    strength_table = np.zeros(len(pair_table))
    for j in range(1, len(pair_table)):
        if pair_table[j] > 0:
            for i in range(1, 3):
                if j - i >= 1:
                    strength_table[j - i] += 3 - i
                if j + i < len(pair_table):
                    strength_table[j + i] += 3 - i
        elif mismatch_table[j] > 0:
            up = j
            while up < len(pair_table) and pair_table[up] == 0:
                up += 1
            down = j
            while down >= 1 and pair_table[down] == 0:
                down -= 1
            if up == len(pair_table) or down < 1:
                minus = 4
            elif pair_table[down] > down and pair_table[up] < up:  # hairpin
                minus = 3
            elif pair_table[down] < down and pair_table[up] > up:  # multi-loop
                minus = 4
            elif pair_table[down] - pair_table[up] == 1:
                minus = 5 if up - down == 2 else 4
            else:
                minus = 3 if up - down == 2 else 2
            for i in range(1, minus):
                if j - i >= 1:
                    strength_table[j - i] -= minus - i
                if j + i < len(pair_table):
                    strength_table[j + i] -= minus - i

    # normalize min of strength map to be one.
    strength_table[1:] -= np.min(strength_table[1:]) - 1
    return strength_table


def _classify(
        pair_table: Sequence[int],
        mismatch_table: Sequence[int],
        junction_table: Sequence[int]):
    """
    Pre-compute the class index for each position.
    Some class might require further sequence specific testing at runtime.
    We put testing lambda function at these positions, which takes in sequence as input and return class index.

    TODO: we need to document what's the meaning / classification criterion for each class.

    :param pair_table: pair index table.
    :param mismatch_table: pre-computed mismatch table from pair_table. See _make_mismatch_table for more info.
    :param junction_table: pre-computed junction table. See _make_junction_table for more info.
    :return: list of class index or lambda function to determine index at each position.
    """
    class_indices = list([None] * (len(pair_table) - 1))
    for pos in range(len(pair_table) - 1):
        if pair_table[pos + 1] > 0:  # it's a pair
            class_indices[pos] = -1
            if (pos < len(pair_table) - 2 and mismatch_table[pos + 2] > 0 and
                    0 < pair_table[pos + 2] < len(pair_table) - 1 and mismatch_table[
                        pair_table[pos + 2] + 1] == 1 + pos):
                if junction_table[pos + 1] > 0 or junction_table[pair_table[pos + 1]] > 0:
                    class_indices[pos] = 0
            if (pos > 0 and mismatch_table[pos] > 0 and pair_table[pos] > 1
                    and mismatch_table[pair_table[pos] - 1] == pos + 1):
                if junction_table[pos + 1] > 0 or junction_table[pair_table[pos + 1]] > 0:
                    class_indices[pos] = 1
            # Tuning for tri-loops
            # Only GC/CG closing pairs work if the supporting pair also is GC/CG
            if pair_table[pos + 1] > pos + 1:
                d = pair_table[pos + 1] - (pos + 1)
                j = pos + 2
                k = pos
            else:
                d = pos + 1 - pair_table[pos + 1]
                j = pair_table[pos + 1] + 1
                k = pair_table[pos + 1] - 1

            if d == 4:
                # pair_values = {'GC': 3, 'CG': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1}
                def _classify_class_2(sequence, k_val=k, pk_val=pair_table[k]):
                    # TODO: better define function name to reflect the functionality
                    if sequence[k_val - 1] + sequence[pk_val - 1] in {'CG', 'GC'}:
                        return 2
                    return -1

                class_indices[pos] = _classify_class_2
            if d == 6 and pair_table[j] > 0 and pair_table[j] - j == 4:
                def _classify_class_3(sequence, j_val=j, pj_val=pair_table[j]):
                    # TODO: better define function name to reflect the functionality
                    if (sequence[j_val - 1] not in 'AUGC' and
                            sequence[j_val - 1] + sequence[pj_val - 1] not in {'CG', 'GC'}):
                        return 3
                    return -1

                class_indices[pos] = _classify_class_3
        else:
            class_indices[pos] = -2
            if mismatch_table[pos + 1] > 0:
                if pair_table[mismatch_table[pos + 1]]:  # mismatch is paired
                    class_indices[pos] = lambda sequence, mp=mismatch_table[pos + 1]: \
                        17 + ['A', 'U', 'G', 'C'].index(sequence[mp - 1])
                else:  # mismatch is unpaired
                    if pos > 0 and pair_table[pos] == mismatch_table[pos + 1] + 1:
                        up, down = pos + 1, mismatch_table[pos + 1]
                    else:
                        up, down = mismatch_table[pos + 1], pos + 1
                    close = up - 1
                    u, d = 0, 0
                    while up + u < len(pair_table) and pair_table[up + u] == 0:
                        u += 1
                    while down > d and pair_table[down - d] == 0:
                        d += 1
                    internal_loop = up + u < len(pair_table) and down > d and pair_table[up + u] == down - d
                    if internal_loop and down - d == close:  # hairpin
                        # A test for a potential 'slide' from a triloop to a GAAA tetraloop
                        # If matched, have a good chance to apply the "anti-boost" (U/C in the middle
                        # of the triloop)
                        if pos > 1 and d == 3:
                            def _classify_class_5(sequence, p=pos):
                                if (sequence[p - 1] == 'G' and sequence[p + 1] not in 'AUGC'
                                        and sequence[p - 2] + sequence[p + 3] in {'CG', 'GC', 'AU', 'UA', 'GU', 'UG'}
                                        and np.random.rand() > 0.5):
                                    return 5
                                return -2
                            class_indices[pos] = _classify_class_5

                        # Simple apical loop G/A boosting
                        if d > 3:
                            def _classify_class_6(_):
                                if np.random.randint(6) > 0:
                                    return 6
                                else:
                                    return -2
                            class_indices[pos] = _classify_class_6
                    elif internal_loop:
                        def _classify_internal_loop(sequence, p=pos, mp=mismatch_table[pos + 1], u_val=u, d_val=d):
                            if sequence[mp - 1] not in 'AUGC':
                                # 1-1 internal loop
                                if u_val == 1 and d_val == 1 and np.random.randint(10) < 8:
                                    return 7
                                elif u_val == 2 and d_val == 2:  # 2-2 internal loop
                                    if (sequence[p + 1] not in 'AUGC' and sequence[mp - 2] not in 'AUGC'
                                            and np.random.rand() > 0.5):
                                        return 8
                                # if a selection hasn't been made yet, try typical boosts for
                                # internal loops (G/A, A/G, U/U)
                                if np.random.rand() < 0.3:
                                    return 9
                                elif mp > p + 1:
                                    return 21
                                else:
                                    return 22
                            else:
                                # Class index is 10, 11, 12, 13 for 'AUGC' respectively
                                return 10 + 'AUGC'.index(sequence[mp - 1])

                        class_indices[pos] = _classify_internal_loop
                    else:  # a mismatch in a junction or external loop
                        if down == pos + 1:
                            if not (pos < 1 or pair_table[pos] == 0):
                                class_indices[pos] = 14
                            else:
                                def _classify_class_15(sequence, p=pos, pp=pair_table[pos + 2]):
                                    if sequence[p + 1] == 'G' and sequence[pp - 1] == 'C':
                                        return 15
                                    return 14
                                class_indices[pos] = _classify_class_15
                        else:
                            if not (pos >= len(pair_table) - 2 or pair_table[pos + 2] == 0):
                                class_indices[pos] = 14
                            else:
                                def _classify_class_16(sequence, p=pos, pp=pair_table[pos]):
                                    if sequence[p - 1] == 'G' and sequence[pp - 1] == 'C':
                                        return 16
                                    return 14
                                class_indices[pos] = _classify_class_16
    return class_indices


def _play_move(
        sequence: list,
        pair_table: Sequence[int],
        mismatch_table: Sequence[int],
        junction_table: Sequence[int],
        bases: str = "AUGC",
        is_debug: bool = False,
        weight_prob: dict = None):
    """
    Update one position / pair in un-allocated positions in sequence.

    :param sequence: the RNA sequence to be filled as list of chars
    :param pair_table: the pair table list of the paired index of each position.
    :param mismatch_table: the mismatch table created by _make_mismatch_table
    :param junction_table: the junction table created by _make_junction_table
    :param bases: the bases that considered as allocated
    :param is_debug: whether to return debug info (class index, etc.)
    :param weight_prob: probabilities for random sampling for different cases.
    :return: the updated RNA sequence and debug info if is_debug is enabled.
    """
    pos = 0
    for pos, ch in enumerate(sequence):
        if ch not in bases:
            break
    debug_info = {"class_index": [], "pos": pos}
    if pair_table[pos + 1] > 0:  # it's a pair
        weight = weight_prob[-1] if weight_prob is not None else np.array([16, 16, 9, 9, 2, 2])
        # Tuning of weights for adjacent stacks in junctions.
        # When looking at adjacent stacks in junctions from the point of view of an
        # observer at the center of the loop, it is usually better to make sure that
        # the left one has a high probability of being GC/CG, while the rightmost one
        # can usually afford to be demoted to AU/UA.
        if (pos < len(pair_table) - 2 and mismatch_table[pos + 2] > 0 and
                0 < pair_table[pos + 2] < len(pair_table) - 1 and mismatch_table[pair_table[pos + 2] + 1] == 1 + pos):
            # possibly adjacent stacks, but we need to eliminate the possibility
            # that it could be a 0-N bulge
            if junction_table[pos + 1] > 0 or junction_table[pair_table[pos + 1]] > 0:
                debug_info['class_index'].append(0)
                if weight_prob is not None:
                    weight = weight_prob[0]
                else:
                    weight[:4] += [-6, -6, 6, 6]
        if (pos > 0 and mismatch_table[pos] > 0 and pair_table[pos] > 1
                and mismatch_table[pair_table[pos] - 1] == pos + 1):
            if junction_table[pos + 1] > 0 or junction_table[pair_table[pos + 1]] > 0:
                debug_info['class_index'].append(1)
                if weight_prob is not None:
                    weight = weight_prob[1]
                else:
                    weight[:4] += [6, 6, -6, -6]
        # Tuning for tri-loops
        # Only GC/CG closing pairs work if the supporting pair also is GC/CG
        if pair_table[pos + 1] > pos + 1:
            d = pair_table[pos + 1] - (pos + 1)
            j = pos + 2
            k = pos
        else:
            d = pos + 1 - pair_table[pos + 1]
            j = pair_table[pos + 1] + 1
            k = pair_table[pos + 1] - 1
        pair_values = {'GC': 3, 'CG': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1}
        if d == 4 and pair_values.get(sequence[k - 1] + sequence[pair_table[k] - 1], 0) == 3:
            debug_info['class_index'].append(2)
            if weight_prob is not None:
                weight = weight_prob[2]
            else:
                weight[2:] = 0
        if (d == 6 and pair_table[j] > 0 and pair_table[j] - j == 4 and sequence[j - 1] not in 'AUGC'
                and pair_values.get(sequence[j - 1] + sequence[pair_table[j] - 1], 0) != 3):
            debug_info['class_index'].append(3)
            if weight_prob is not None:
                weight = weight_prob[3]
            else:
                weight[:2] = 0

        # Select according to the weights
        sequence[pos], sequence[pair_table[pos + 1] - 1] = np.random.choice(
            ['CG', 'GC', 'UA', 'AU', 'UG', 'GU'], p=weight / np.sum(weight))
        if len(debug_info['class_index']) == 0:
            debug_info['class_index'].append(-1)

    else:  # unpaired, choose a random base
        if weight_prob is not None:
            weight = weight_prob[-2]
        else:
            weight = np.array([93, 1, 5, 1])
        if mismatch_table[pos + 1] > 0:
            if pair_table[mismatch_table[pos + 1]]:  # mismatch is paired
                # Use 17, 18, 19, 20 for 'A', 'U', 'G', 'C' respectively.
                ch = sequence[mismatch_table[pos + 1] - 1]
                class_index = 17 + ['A', 'U', 'G', 'C'].index(ch)
                debug_info['class_index'].append(class_index)
                if weight_prob is not None:
                    weight = weight_prob[class_index]
                else:
                    weight = {'A': [5, 0, 2, 1], 'U': [0, 6, 1, 4], 'G': [2, 1, 5, 0], 'C': [6, 4, 0, 1]}[ch]
            else:  # mismatch is unpaired
                if pos > 0 and pair_table[pos] == mismatch_table[pos + 1] + 1:
                    up, down = pos + 1, mismatch_table[pos + 1]
                else:
                    up, down = mismatch_table[pos + 1], pos + 1
                close = up - 1
                u, d = 0, 0
                while up + u < len(pair_table) and pair_table[up + u] == 0:
                    u += 1
                while down > d and pair_table[down - d] == 0:
                    d += 1
                internal_loop = up + u < len(pair_table) and down > d and pair_table[up + u] == down - d
                if internal_loop and down - d == close:  # hairpin
                    # A test for a potential 'slide' from a triloop to a GAAA tetraloop
                    # If matched, have a good chance to apply the "anti-boost" (U/C in the middle
                    # of the triloop)
                    pair_values = {'GC': 3, 'CG': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1}
                    if (pos > 1 and d == 3 and sequence[pos + 1] not in 'AUGC'
                            and pair_values.get(sequence[pos - 2] + sequence[pos + 3], 0) > 0
                            and sequence[pos - 1] == 'G' and np.random.rand() > 0.5):
                        debug_info['class_index'].append(5)
                        dice = np.random.randint(10)
                        if dice < 5:
                            sequence[pos + 1] = 'U'
                        elif dice < 9:
                            sequence[pos + 1] = 'C'

                    # Simple apical loop G/A boosting
                    if sequence[mismatch_table[pos + 1] - 1] not in 'AUGC' and d > 3:
                        debug_info['class_index'].append(6)
                        to_update = (np.random.rand() < weight_prob[6][2] if weight_prob is not None
                                     else np.random.randint(6) > 0)
                        if to_update:
                            sequence[mismatch_table[pos + 1] - 1] = 'A'
                            sequence[pos] = 'G'
                elif internal_loop:
                    weight = np.array([5, 1, 20, 1]) if mismatch_table[pos + 1] > pos + 1 else np.array([21, 1, 4, 1])
                    if sequence[mismatch_table[pos + 1] - 1] not in 'AUGC':
                        # 1-1 internal loop
                        if u == 1 and d == 1:
                            debug_info['class_index'].append(7)
                            to_update = (np.random.rand() < weight_prob[7][2] if weight_prob is not None
                                         else np.random.randint(10) < 8)
                            if to_update:
                                sequence[mismatch_table[pos + 1] - 1] = 'G'
                                sequence[pos] = 'G'
                        elif u == 2 and d == 2:  # 2-2 internal loop
                            if (sequence[pos + 1] not in 'AUGC'
                                    and sequence[mismatch_table[pos + 1] - 2] not in 'AUGC'):
                                debug_info['class_index'].append(8)
                                to_update = (np.random.rand() < weight_prob[8][1] if weight_prob is not None
                                             else np.random.rand() > 0.5)
                                if to_update:
                                    sequence[mismatch_table[pos + 1] - 1] = 'G'
                                    sequence[pos + 1] = 'G'
                                    sequence[mismatch_table[pos + 1] - 2] = 'U'
                                    sequence[pos] = 'U'
                        # if a selection hasn't been made yet, try typical boosts for
                        # internal loops (G/A, A/G, U/U)
                        if sequence[pos] not in 'AUGC':
                            debug_info['class_index'].append(9)
                            if np.random.rand() < 0.3:
                                prob = (weight_prob[9][[2, 0, 1]] if weight_prob is not None
                                        else np.array([3, 3, 1]))
                                sequence[pos], sequence[mismatch_table[pos + 1] - 1] = np.random.choice(
                                    ['GA', 'AG', 'UU'], p=prob / np.sum(prob))
                    else:
                        # Class index is 10, 11, 12, 13 for 'AUGC' respectively
                        debug_info['class_index'].append(
                            10 + 'AUGC'.index(sequence[mismatch_table[pos + 1] - 1]))
                        if weight_prob is not None:
                            weight = np.array(
                                {
                                    'A': weight_prob[10],
                                    'U': weight_prob[11],
                                    'G': weight_prob[12],
                                    'C': weight_prob[13],
                                }[sequence[mismatch_table[pos + 1] - 1]])
                        else:
                            weight = np.array(
                                {
                                    'A': [4, 0, 4, 1],
                                    'U': [0, 6, 1, 2],
                                    'G': [6, 1, 2, 0],
                                    'C': [4, 1, 0, 1]
                                }[sequence[mismatch_table[pos + 1] - 1]])
                else:  # a mismatch in a junction or external loop
                    weight = weight_prob[14] if weight_prob is not None else np.array([97, 1, 1, 1])
                    debug_info['class_index'].append(14)
                    if down == pos + 1:
                        if ((pos < 1 or pair_table[pos] == 0) and sequence[pos + 1] == 'G'
                                and sequence[pair_table[pos + 2] - 1] == 'C'):
                            debug_info['class_index'].remove(14)
                            debug_info['class_index'].append(15)
                            if weight_prob is not None:
                                weight = weight_prob[15]
                            else:
                                weight[3] = 48  # C boost
                    else:
                        if ((pos >= len(pair_table) - 2 or pair_table[pos + 2] == 0) and sequence[pos - 1] == 'G'
                                and sequence[pair_table[pos] - 1] == 'C'):
                            debug_info['class_index'].remove(14)
                            debug_info['class_index'].append(16)
                            if weight_prob is not None:
                                weight = weight_prob[16]
                            else:
                                weight[2] = 48
        if sequence[pos] not in 'AUGC':
            sequence[pos] = np.random.choice(['A', 'U', 'G', 'C'], p=weight / np.sum(weight))
        if len(debug_info['class_index']) == 0:
            debug_info['class_index'].append(-2)
    if is_debug:
        return sequence, debug_info
    else:
        return sequence


@profile
def _sample_with_class_indices(sequence: list,
                               pair_table: Sequence[int],
                               mismatch_table: Sequence[int],
                               class_indices: list):
    """
    Fill the sequence undecided positions with random samples.
    The sampling weight / probabilities is different in each class.

    :param sequence: the RNA sequence to be filled.
    :param pair_table: the pair table list of the paired index of each position.
    :param class_indices: the pre-computed class indices.
    :return: the filled RNA sequence.
    """
    sequence = list(sequence)

    # Fill in paired positions
    for index, ch in enumerate(sequence):
        if ch != 'N' or pair_table[index + 1] == 0:
            continue
        class_index = class_indices[index]
        if callable(class_index):
            class_index = class_index(sequence)
        weight = np.array([16, 16, 9, 9, 2, 2])
        if class_index == 0:
            weight = np.array([10, 10, 15, 15, 2, 2])
        elif class_index == 1:
            weight = np.array([22, 22, 3, 3, 2, 2])
        elif class_index == 2:
            weight[2:] = 0
        elif class_index == 3:
            weight[:2] = 0
        sequence[index], sequence[pair_table[index + 1] - 1] = np.random.choice(
            ['CG', 'GC', 'UA', 'AU', 'UG', 'GU'], p=weight / np.sum(weight))

    # Fill in un-paired positions
    for index, ch in enumerate(sequence):
        if ch != 'N':
            continue
        class_index = class_indices[index]
        if callable(class_index):
            class_index = class_index(sequence)
        weight = np.array([93, 1, 5, 1])
        if 17 <= class_index <= 20:
            ch = sequence[mismatch_table[index + 1] - 1]
            weight = {'A': [5, 0, 2, 1], 'U': [0, 6, 1, 4], 'G': [2, 1, 5, 0], 'C': [6, 4, 0, 1]}[ch]
        elif class_index == 5:
            dice = np.random.randint(10)
            if dice < 5:
                sequence[index + 1] = 'U'
            elif dice < 9:
                sequence[index + 1] = 'C'
        elif class_index == 6:
            sequence[mismatch_table[index + 1] - 1] = 'A'
            sequence[index] = 'G'
        elif class_index == 7:
            sequence[mismatch_table[index + 1] - 1] = 'G'
            sequence[index] = 'G'
        elif class_index == 8:
            sequence[mismatch_table[index + 1] - 1] = 'G'
            sequence[index + 1] = 'G'
            sequence[mismatch_table[index + 1] - 2] = 'U'
            sequence[index] = 'U'
        elif class_index == 9:
            prob = np.array([3, 3, 0.1])
            sequence[index], sequence[mismatch_table[index + 1] - 1] = np.random.choice(
                ['GA', 'AG', 'UU'], p=prob / np.sum(prob))
        elif 10 <= class_index <= 13:
            weight = np.array(
                {
                    'A': [4, 0, 4, 1],
                    'U': [0, 6, 1, 2],
                    'G': [6, 1, 2, 0],
                    'C': [4, 1, 0, 1]
                }[sequence[mismatch_table[index + 1] - 1]])
        elif class_index == 14:
            weight = np.array([97, 1, 1, 1])
        elif class_index == 15:
            weight = np.array([97, 1, 1, 48])
        elif class_index == 16:
            weight = np.array([97, 1, 48, 1])
        elif class_index == 21:
            weight = np.array([5, 1, 20, 1])
        elif class_index == 22:
            weight = np.array([21, 1, 4, 1])
        if sequence[index] == 'N':
            sequence[index] = np.random.choice(['A', 'U', 'G', 'C'], p=weight / np.sum(weight))
    return sequence


def _sample(sequence: list,
            pair_table: Sequence[int],
            mismatch_table: Sequence[int],
            junction_table: Sequence[int],
            weight_prob: dict = None):
    """
    Fill the sequence undecided positions with play_move strategy.

    :param sequence: the RNA sequence to be filled.
    :param pair_table: the pair table list of the paired index of each position.
    :return: the filled RNA sequence.
    """
    for index, ch in enumerate(sequence):
        if ch == 'N' and pair_table[index + 1] > 0:
            sequence[index] = 'P'
    # Fill in paired positions
    while not (set(sequence) <= {'A', 'U', 'G', 'C', 'N'}):
        sequence = _play_move(sequence, pair_table, mismatch_table, junction_table, 'AUGCN', weight_prob=weight_prob)

    # Fill the unpaired bases
    while not (set(sequence) <= {'A', 'U', 'G', 'C'}):
        sequence = _play_move(sequence, pair_table, mismatch_table, junction_table, 'AUGC', weight_prob=weight_prob)
    return sequence

@profile
def _fill_bases(sequence: list,
                pair_table: Sequence[int],
                pos: int,
                bases_index):
    """
    Fill a bases at given position to selected type.

    :param sequence: the RNA sequence to be filled.
    :param pair_table: the pair table list of the paired index of each position.
    :param pos: position to be filled.
    :param bases_index: the index to bases to be filled.
    :return: the filled RNA sequence.
    """
    sequence = list(sequence)
    if pair_table[pos + 1] == 0:
        sequence[pos] = 'AUGC'[bases_index]
    else:
        sequence[pos], sequence[pair_table[pos + 1] - 1] = (
            ['CG', 'GC', 'UA', 'AU', 'UG', 'GU'][bases_index])
    return sequence


def _unset_bases(sequence: list,
                 pair_table: Sequence[int],
                 pos: int):
    """
    Unset bases at given location back to N

    :param sequence: the RNA sequence to be filled.
    :param pair_table: the pair table list of the paired index of each position.
    :param pos: position to be filled.
    :return: the processed RNA sequence.
    """
    sequence[pos] = 'N'
    if pair_table[pos + 1] > 0:
        sequence[pair_table[pos + 1] - 1] = 'N'
    return sequence


@profile
def _fill_seq(target_fold: str,
              sequence: list,
              pair_table: Sequence[int],
              mismatch_table: Sequence[int],
              junction_table: Sequence[int],
              weight_prob: dict = None,
              class_indices: list = None):
    """
    Fill the RNA sequence locations with unknown value (e.g. N)
    <More description about the rule here>

    :param sequence: the RNA sequence to be filled.
    :param pair_table: the pair table list of the paired index of each position.
    :return: the filled RNA sequence.
    """
    if set(sequence) <= {'A', 'U', 'G', 'C'}:  # already fully filled
        return sequence

    sequence = sequence.copy()  # make a local copy so we don't mess up with input
    global_best, local_best, best_sequence, solved = -np.inf, '', '', False
    while not (set(sequence) <= {'A', 'U', 'G', 'C'}):
        pos, max_score = 0, -np.inf
        for pos, ch in enumerate(sequence):
            if ch not in 'AUGC':
                break
        n_choice = 4 if pair_table[pos + 1] == 0 else 6
        for bases_index in range(n_choice):
            local_sequence = _fill_bases(sequence, pair_table, pos, bases_index)
            if class_indices is not None:
                local_sequence = _sample_with_class_indices(local_sequence, pair_table, mismatch_table, class_indices)
            else:
                local_sequence = _sample(local_sequence, pair_table, mismatch_table, junction_table,
                                         weight_prob=weight_prob)
            local_score, solved = compute_score(target_fold, ''.join(local_sequence))
            if local_score > max_score or solved:
                max_score, local_best = local_score, local_sequence
            if solved:
                break
        if max_score > global_best or solved:
            global_best, best_sequence = max_score, local_best
        if solved:
            return best_sequence
        sequence[pos] = best_sequence[pos]
        if pair_table[pos + 1] > 0:
            sequence[pair_table[pos + 1] - 1] = best_sequence[pair_table[pos + 1] - 1]
        elif (mismatch_table[pos + 1] > 0 and sequence[mismatch_table[pos + 1] - 1] == 'N'
              and pair_table[mismatch_table[pos + 1] - 1] == 0):
            sequence[mismatch_table[pos + 1] - 1] = best_sequence[mismatch_table[pos + 1] - 1]
    return best_sequence


def _print(message: str, color: str = None):
    """
    Print a colored message.

    :param message: the string to be printed.
    :param color: color to be used, now only support
                  red, green, blue, yellow, magenta, cyan
    """
    color_map = {
        "red": "\u001b[31;1m",
        "green": "\u001b[32;1m",
        "yellow": "\u001b[33;1m",
        "blue": "\u001b[34;1m",
        "magenta": "\u001b[35;1m",
        "cyan": "\u001b[36;1m"
    }
    reset = "\033[0m"

    if color is None or platform.system() == "Windows" or color not in color_map:
        print(message)
    else:
        print(color_map[color] + message + reset)


def _run_eterna100(weights=None):
    """
    Run Nemo on Eterna 100 benchmark data set.
    """
    eterna_data_path = os.path.join(
        os.path.dirname(__file__), '../data/list.e100')
    with open(eterna_data_path, 'r') as f:
        raw_data = f.read().splitlines()
        puzzle_data = []
        for row in raw_data:
            puzzle_info = [x.strip() for x in row.split(" ")]
            puzzle_info[0] = puzzle_info[0][1:-1]  # remove quotation marks
            if len(puzzle_info) == 1:
                puzzle_info.append('N' * len(puzzle_info[0]))
            puzzle_data.append(puzzle_info)

    solved_count = 0
    test_start_index = 63
    test_indices = [63]
    for test_index, (test_target_fold, test_start_seq) in enumerate(puzzle_data):
        if test_index < test_start_index or test_index not in test_indices:
            continue
        print('Solving problem {}/{}....'.format(test_index, len(puzzle_data)))
        print('....target fold: {}'.format(test_target_fold))
        print('....start sequence: {}'.format(test_start_seq))
        test_sequence = rna_inverse(test_target_fold, test_start_seq, weight_prob=weights)
        print_perf_data()
        clear_perf_data()
        if test_sequence is not None:
            solved_count += 1
            _print('Test case {} solved: {}'.format(test_index, test_sequence), color='green')
        else:
            _print('Test case {} failed: {}'.format(test_index, test_target_fold), color='red')
    _print('Solved {}/{} puzzles'.format(solved_count, len(puzzle_data)), color='green')
    return solved_count


def _weight_probability_fit():
    """
    Fit sampling weight / probability for each class on eterna_complete data set.
    """
    eterna_data_path = os.path.join(
        os.path.dirname(__file__),
        '../third_party/SentRNA/data/train/eterna_complete_ss.txt')
    stats = dict()
    with open(eterna_data_path, 'r') as f:
        for row_index, row in enumerate(f):
            if (row_index + 1) % 10000 == 0:
                print('Processed {} rows....'.format(row_index + 1))
            puzzle_info = [x.strip() for x in row.split("    ")]
            if len(puzzle_info) == 2:
                continue  # skip rows without answer
            puzzle_name, target_fold, solution, locks = puzzle_info

            # Generate start sequence.
            sequence = np.array(list('N' * len(target_fold)))
            locked_indices = np.array(list(locks)) == 'x'
            sequence[locked_indices] = np.array(list(solution))[locked_indices]

            # Make tables
            pair_table = RNA.ptable(target_fold)
            mismatch_table = _make_mismatch_table(pair_table)
            loop_table = _make_loop_table(pair_table)
            junction_table = _make_junction_table(pair_table, loop_table)

            # Fill in the first unknown position.
            for index, ch in enumerate(sequence):
                if ch == 'N':
                    sequence[index] = solution[index]
                    if pair_table[index + 1] > 0:
                        sequence[pair_table[index + 1] - 1] = solution[pair_table[index + 1] - 1]
                    break

            # Fill in paired locations.
            for index, ch in enumerate(sequence):
                if ch == 'N' and pair_table[index + 1] > 0:
                    sequence[index] = 'P'
            while not (set(sequence) <= {'A', 'U', 'G', 'C', 'N'}):
                nemo_sequence, debug_info = _play_move(
                    sequence.copy(), pair_table, mismatch_table, junction_table,
                    'AUGCN', is_debug=True)
                if len(debug_info['class_index']) == 1:
                    if debug_info['class_index'][0] not in stats:
                        stats[debug_info['class_index'][0]] = []
                    stats[debug_info['class_index'][0]].append(
                        solution[debug_info['pos']] + solution[pair_table[debug_info['pos'] + 1] - 1])

                # Update all changed positions to the values in solution.
                changed_positions = (sequence != nemo_sequence)
                sequence[changed_positions] = np.array(list(solution))[changed_positions]

            # Fill in unpaired locations.
            while not (set(sequence) <= {'A', 'U', 'G', 'C'}):
                nemo_sequence, debug_info = _play_move(
                    sequence.copy(), pair_table, mismatch_table, junction_table,
                    'AUGC', is_debug=True)
                if len(debug_info['class_index']) == 1:
                    if debug_info['class_index'][0] not in stats:
                        stats[debug_info['class_index'][0]] = []
                    stats[debug_info['class_index'][0]].append(solution[debug_info['pos']])

                # Update all changed positions to the values in solution.
                changed_positions = (sequence != nemo_sequence)
                sequence[changed_positions] = np.array(list(solution))[changed_positions]

    # Convert stats to probabilities
    probabilities = dict()
    for class_stats_index in stats:
        val, count = np.unique(stats[class_stats_index], return_counts=True)
        prob = np.array(count) / np.sum(count)
        if len(val[0]) == 2:
            probabilities[class_stats_index] = np.zeros(6)
            for i, bases_type in enumerate(['CG', 'GC', 'UA', 'AU', 'UG', 'GU']):
                if bases_type in val:
                    probabilities[class_stats_index][i] = prob[list(val).index(bases_type)]
        elif len(val[0]) == 1:
            probabilities[class_stats_index] = np.zeros(4)
            for i, bases_type in enumerate(['A', 'U', 'G', 'C']):
                if bases_type in val:
                    probabilities[class_stats_index][i] = prob[list(val).index(bases_type)]
        else:
            print('Unknown probability format for type: {}'.format(val))

    for class_index in probabilities:
        print('Class {}: {}'.format(class_index, probabilities[class_index]))
    return stats, probabilities


if __name__ == '__main__':
    # Configure Vienna RNA
    RNA.read_parameter_file(os.path.join(
        os.path.dirname(__file__), '../data/vrna185x.par'))

    # _run_eterna100()
    # _, weight_probabilities = _weight_probability_fit()
    # _run_eterna100(weight_probabilities)

    global_start_time = time.time()
    _run_eterna100()
    print('Finished in {}'.format(time.time() - global_start_time))
