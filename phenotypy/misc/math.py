import numpy as np


def largest_factor(n):

    for i in range(2, int(np.floor(np.sqrt(n))) + 1):

        if n % i == 0:
            return n // i

        return None  # prime
