import numpy as np
import matplotlib.pyplot as plt
from trackhhl.event_model.event_model import Segment

def find_hit_pairs(event,solution):
        """
        Finds and returns hit pairs from an event based on a given solution.
        Args:
            event (Event): An event object containing hits with attributes including 'z'.
            solution (np.ndarray): A numpy array representing the solution with non-zero indices.
        Returns:
            tuple: A tuple containing:
                - segments (list of Segment): A list of Segment objects created from the hit pairs.
                - layers (list of str): A list of layer identifiers as strings.
                - H (list of int): A list of hit counts per layer.
                - P (np.ndarray): A numpy array representing the cumulative hit pair counts.
        The function performs the following steps:
            1. Groups hits by their 'z' coordinate into a dictionary `hits_on_layer`.
            2. Counts the number of hits per layer and stores it in `hits_per_layer`.
            3. Initializes lists `layers` and `H` to store layer identifiers and hit counts respectively.
            4. Computes the cumulative hit pair counts `P`.
            5. Identifies non-zero indices in the solution.
            6. Iterates through the non-zero indices to find corresponding hit pairs.
            7. Creates Segment objects from the hit pairs and returns them along with `layers`, `H`, and `P`.
        """
    
    hits_on_layer = {f'{int(z)}' : [h for h in event.hits if h.z == z] for z in set([h.z for h in event.hits])}
    hits_per_layer = {f'{int(z)}' : len(hits_on_layer[z]) for z in hits_on_layer.keys()}

    #  To keep in memory: 
    #  H : list[int], P : list[int],non_zero_indices : list[int]
    

    layers = list(hits_per_layer.keys())
    H = list(hits_per_layer.values())
  

    P = np.zeros(len(layers))
    for i in range(1, len(layers)):
        count = 0
        for j in range(i):
            count += H[j] * H[j + 1]
        P[i] = count



    non_zero_indices = np.nonzero(solution)[0]
    # print(f'non_zero_indices: {non_zero_indices}')

    hit_pairs = []
    print(P)
    for eta in non_zero_indices:
        for j in range(0, P.shape[0]):
            if P[j] <= eta < P[j + 1]:
                tau = eta - P[j]
                alpha, beta = divmod(tau, H[j + 1])
                # print(f'eta: {eta}, j: {j}, tau: {tau}, alpha: {alpha}, beta: {beta}, layers: {layers[j]}, {layers[j + 1]}')
                hit_pairs.append((hits_on_layer[f'{layers[int(j)]}'][int(alpha)], hits_on_layer[f'{layers[int(j + 1)]}'][int(beta)]))

    segments = []
    for i, hit_pair in enumerate(hit_pairs):
        segments.append(Segment(segment_id = i, hit_from = hit_pair[0], hit_to = hit_pair[1]))

    return segments, layers, H, P

if __name__ == "__main__":
    # Replace `event` and `solution` with actual values or function calls
    event = ...  # Define or load your event object
    solution = ...  # Define or load your solution array
    segments, layers, H, P = find_hit_pairs(event, solution)
    print(segments, layers, H, P)

