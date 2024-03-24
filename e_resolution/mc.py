import numpy as np
import scipy.stats as stats

# Constants
NUM_TILES = 10  # Number of SiPM tiles
CHANNELS_PER_TILE = 16  # Number of channels per SiPM tile
N = 100  # Number of tests
PHOTONS_PER_TEST = 4000  # Number of photons simulated in each test

# Example parameters
p_active = 0.9  # Probability of photon hitting SiPM active area
pde_channels = np.random.rand(NUM_TILES, CHANNELS_PER_TILE)  # Photon detection efficiency per channel
p_ap = 0.03  # Afterpulsing probability
optical_crosstalk_param = 1.2  # Parameter for generalized Poisson distribution

def generalized_poisson(lmbda, xi):
    """ Generate a random number from a generalized Poisson distribution. """
    return stats.poisson.rvs(mu=lmbda + xi * np.random.poisson(lmbda))

def simulate_photon():
    """ Simulate the journey of a single photon. """
    # Check if photon hits the active area
    if np.random.rand() > p_active:
        return 0, 0, 0

    # Select a random tile and channel
    tile = np.random.randint(NUM_TILES)
    channel = np.random.randint(CHANNELS_PER_TILE)

    # Check if photon is detected
    if np.random.rand() > pde_channels[tile, channel]:
        return 1, 0, 0

    # Simulate optical crosstalk
    electrons = generalized_poisson(1, optical_crosstalk_param)

    # Simulate afterpulsing
    afterpulses = np.sum(np.random.rand(electrons) < p_ap)

    return 1, electrons, electrons + afterpulses

# Running N tests
all_test_results = []

for _ in range(N):
    test_results = [simulate_photon() for _ in range(PHOTONS_PER_TEST)]
    initial_photons = sum(r[0] for r in test_results)
    detected_electrons = sum(r[1] for r in test_results)
    total_electrons = sum(r[2] for r in test_results)
    all_test_results.append((initial_photons, detected_electrons, total_electrons))

# Example: Analyzing results of the first test
first_test = all_test_results[0]
print(f"First Test - Initial Photons: {first_test[0]}, Detected Electrons: {first_test[1]}, Total Electrons: {first_test[2]}")

# Further analysis can be performed on all_test_results as needed

