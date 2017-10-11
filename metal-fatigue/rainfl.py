import numpy as np

# definition of a class which represents a rainflow matrix


class rfm:
    def __init__(self, xbinsize, ybinsize, counts):
        self.xbinsize = xbinsize
        self.ybinsize = ybinsize
        # counts should be a numpy array
        self.counts = counts
