# Adapted from https://github.com/QData/dmc_remastered (Apache-2.0).
# Modified for DRP: local package imports and formatting. See LICENSE in this directory.

import random


class dmcr_random:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.old_state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *args, **kwargs):
        random.setstate(self.old_state)
