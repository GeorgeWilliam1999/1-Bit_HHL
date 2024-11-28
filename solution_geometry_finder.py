import numpy as np

def find_hit_pairs(events: list, solution_vector: np.array):
    """
    Finds the geometry of the segments of numerous particle tracks.

    Parameters:
    events (list): A list of event objects. Each event contains hits, where each hit has a 'z' attribute indicating the layer.
    solution_vector (np.array): A binary vector of dimension 2^n indexing all pairs of hits between each layer.
                                It gives a 1 when the segment is present and 0 when the hits are not from the same particle.

    Returns:
    tuple: A tuple containing:
        - hit_pairs (list): A list of tuples, where each tuple contains pairs of hits that form segments of particle tracks.
        - layers (list): A list of layer identifiers.
        - H (list): A list of lists, where each sublist contains hits for a specific layer.
        - P (np.array): An array where each element represents the cumulative count of all possible segments up to that layer.
    """
    hits_per_layer = {}
    for event in events:
        hits_on_layer = {f'{int(z)}': [h for h in event.hits if h.z == z] for z in set([h.z for h in event.hits])}
        hits_per_layer.update(hits_on_layer)

    layers = list(hits_per_layer.keys())
    H = list(hits_per_layer.values())

    # P is the cumulative sum of all possible segments
    P = np.zeros(len(layers))
    for i in range(1, len(layers)):
        count = 0
        for j in range(i):
            count += len(H[j]) * len(H[j + 1])
        P[i] = count

    S = solution_vector
    non_zero_indices = np.nonzero(S)[0]

    hit_pairs = []

    for eta in non_zero_indices:
        for j in range(1, P.shape[0]):
            if P[j] < eta <= P[j + 1]:
                tau = eta - P[j]
                alpha, beta = divmod(tau, len(H[j + 1]))
                hit_pairs.append((hits_on_layer[f'{int(j)}'][int(alpha)], hits_on_layer[f'{int(j + 1)}'][int(beta)]))

    return hit_pairs, layers, H, P