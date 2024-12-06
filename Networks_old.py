import numpy as np
import matplotlib.pyplot as plt

# Define the fragility function for each bridge based on earthquake magnitude M = m
def bridge_failure_probability(Sa, beta):
    return 1 - np.exp(-beta * Sa)

# Example Sa for bridges based on earthquake magnitude
def spectral_acceleration(magnitude):
    # Example function for spectral acceleration based on magnitude
    return 0.5 * magnitude + np.random.normal(0, 0.1, 6) 

# failure probability for a path composed of multiple bridges
def path_failure_probability(bridges_fail_prob):
    # Failure probability of a path is the combined failure of all bridges
    return 1 - np.prod(1 - np.array(bridges_fail_prob))

# Example 
magnitude = 7.0  #earthquake magnitude
beta = 0.76  # fragility parameter
Sa_values = spectral_acceleration(magnitude)

# Paths and their associated bridges (indices of Sa_values)
paths = {
    "Path 1 (5-1)": [0],  
    "Path 2 (5-2-1)": [1, 2],  
    "Path 3 (5-7-1)": [3],  
    "Path 4 (5-2-6-1)": [4, 5],  
}

# failure probabilities 
bridge_failure_probs = [bridge_failure_probability(Sa, beta) for Sa in Sa_values]

# disconnection probabilities for each path
path_failure_probs = {path: path_failure_probability([bridge_failure_probs[i] for i in bridges])
                    for path, bridges in paths.items()}

# Compute the total disconnection event (when all paths fail)
total_disconnection_prob = np.prod(list(path_failure_probs.values()))

plt.barh(list(path_failure_probs.keys()), list(path_failure_probs.values()), color='skyblue')
plt.xlabel('Path Failure Probability')
plt.title(f'Disconnection Probabilities for City 5 (Magnitude = {magnitude})')
plt.grid(True)
plt.show()

print(f"Total disconnection probability: {total_disconnection_prob}")