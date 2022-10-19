"""
SM Harwood
19 October 2022
"""
import sys
import logging
sys.path.append("..")
from formulations.path_based_rp import get_sampled_key as sampler

logging.basicConfig(level=logging.DEBUG)

def test(explore=1):
    """ Test sampler in path_based_rp """
    testD = { 'a':100, 'b':101, 'c':102, 'd':10 }
    counts = { k:0 for k in testD.keys() }
    N = 10000
    for _ in range(N):
        k, _ = sampler(testD, explore)
        counts[k] = counts[k] + 1
    print(counts)

if __name__ == "__main__":
    test()
