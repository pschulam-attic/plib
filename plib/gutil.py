"""
General utilities
"""

def all_same_size(*xs):
    """
    Check if all items passed to the function have the same size. Each
    item must have __len__ implemented.

    """
    if len(xs) == 0:
        return True
    else:
        return all(len(xs[0]) == len(x) for x in xs)

def union_multi_filter(pred, *lists):
    """
    Filter a group of lists using the predicate pred (i.e. a function
    returning True or False). The result is a list of N lists where N
    is the number of lists passed for filtering.
    
    For each list L, the i^th element is kept if the predicate
    evaluates to True for the i^th element of ANY of the N lists.

    """
    if len(lists) == 0:
        return []
    assert all_same_size(*lists)
    filtered = [xs for xs in zip(*lists) if any([pred(x) for x in xs])]
    return zip(*filtered)

def intersect_multi_filter(pred, *lists):
    """
    Filter a group of lists using the predicate pred (i.e. a function
    returning True or False). The result is a list of N lists where N
    is the number of lists passed for filtering.
    
    For each list L, the i^th element is kept if the predicate
    evaluates to True for the i^th element of ALL of the N lists.

    """
    if len(lists) == 0:
        return []
    assert all_same_size(*lists)
    filtered = [xs for xs in zip(*lists) if all([pred(x) for x in xs])]
    return zip(*filtered)
