from unittest import TestCase

import numpy as np
import math


def random_gen(count: int) -> np.ndarray:   # test random generator always puts "[1 2 3...]"
    random_numbers = np.empty(0)
    for i in range(1, count+1):
        random_numbers = np.append(random_numbers, i)
    return random_numbers


class TestCombine_genes_rand_weight(TestCase):

    def test_combine_genes_rand_weight(self):
        from run_evolution import combine_genes_rand_weight
        genes = np.array([3,2,1])

        expected = 1.666666666
        # for 3 genes, the test rng generates [1 2 3], which is normalized to [0.166, 0.333, 0.5]
        # [0.166, 0.333, 0.5]*[3 2 1] = 1.6666

        result = combine_genes_rand_weight(genes, random_gen)
        self.assertTrue(math.isclose(result, expected, rel_tol=1e-5))
