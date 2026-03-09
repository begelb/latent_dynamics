import numpy as np
from scipy.optimize import fsolve

# Model parameters
b = [0, 0, 2.89, 10.03, 21.59, 39.02, 56.41, 77.72, 103.23, 131.87, 164.57, 201.46, 242.65]
survivalRates = [0.889, 0.633, 0.697, 0.517, 0.437, 0.287, 0.571, 0.333, 0.75, 1, 0.333, 1]
surfaceArea = 36

# 1. Calculate Reproductive Value (Phi) and Adult/Larva Ratio (Psi)
phi = b[0]
psi = 0
prod_s = 1.0
for i in range(1, len(b)):
    prod_s *= survivalRates[i-1]
    phi += b[i] * prod_s 
    psi += prod_s        

# 2. Define Equilibrium Condition
def L(rho):
    return 2.94 / (rho + 520 * np.exp(-0.14 * rho))

def find_roots(rho):
    return L(rho) * phi - 1

# 3. Solve for equilibria
unstable_rho = fsolve(find_roots, 15)[0]

def build_vector(rho_star):
    x1 = (rho_star * surfaceArea) / psi
    x = [x1]
    prod_s = 1.0
    for i in range(1, len(b)):
        prod_s *= survivalRates[i-1]
        x.append(x1 * prod_s)
    return np.array(x)

# --- Added Code for Class Distribution ---

# Get the full vector for the unstable point
unstable_vector = build_vector(unstable_rho)

print(f"Unstable Equilibrium (Threshold): Density = {unstable_rho:.2f}")
print("Vector: ", list(unstable_vector))
print(f"Total Population at Threshold: {sum(unstable_vector):.2f}")

print("\nClass Distribution at Unstable Equilibrium:")
print("-" * 45)
print(f"{'Class':<10} | {'Population':<15} | {'Percentage':<10}")
print("-" * 45)

total_pop = sum(unstable_vector)
for i, count in enumerate(unstable_vector):
    percentage = (count / total_pop) * 100
    print(f"Class {i+1:<4} | {count:<15.2f} | {percentage:.2f}%")