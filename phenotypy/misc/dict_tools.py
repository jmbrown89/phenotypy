def reverse_dict(d):
    return {v: k for k, v in d.items()}


def enumerate_dict(iterable, reverse=True):

    d = {k: v for k, v in enumerate(iterable)}

    if reverse:
        return reverse_dict(d)
    return d


