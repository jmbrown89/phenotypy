import numpy as np


def largest_factor(n):

    for i in range(2, int(np.floor(np.sqrt(n))) + 1):

        if n % i == 0:
            return n // i

        return None  # prime


def mrange(start, end, step):

    if isinstance(step, float):

        result = [start]

        for i in range(1, int(np.log(end) / np.log(step)) - int(np.log(start) / np.log(step)) + 1):

            res = result[i-1] * step
            res = int(res) if isinstance(start, int) else float(res)
            result.append(res)

        return result

    else:
        return list(np.arange(start, end, step))
