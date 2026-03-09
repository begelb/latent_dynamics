import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvals

# Import your model logic
from red_coral import corals_model

def find_equilibria(initial_guess, max_iters=2000):
    # 1. Simulation Method: Run the model until it (hopefully) stabilizes
    history = [initial_guess]
    curr_x = np.array(initial_guess, dtype=float)
    
    for i in range(max_iters):
        next_x = corals_model(curr_x)
        if np.allclose(curr_x, next_x, atol=1e-9):
            print(f"Simulation converged at iteration {i}")
            break
        curr_x = next_x
        history.append(curr_x)
    
    sim_equilibrium = curr_x

    # 2. Root-Finding Method: Solve f(x) - x = 0 for precision
    def obj_func(x):
        return corals_model(x) - x

    sol = root(obj_func, sim_equilibrium)
    precise_equilibrium = sol.x if sol.success else None

    return history, sim_equilibrium, precise_equilibrium

# --- Execution ---
initial_pop = [800, 700, 450, 300, 150, 100, 20, 10, 3, 3, 3, 3, 3]  # Starting with a small population in all classes
history, sim_eq, precise_eq = find_equilibria(initial_pop)

print("\n--- Results ---")
print(f"Steady State Population (Total): {sum(precise_eq):.2f}")
print(f"Class Distribution:\n{precise_eq}")

# Optional: Plotting the convergence
plt.figure(figsize=(10, 6))
plt.plot([sum(h) for h in history])
plt.title("Total Population Convergence Over Time")
plt.xlabel("Generations")
plt.ylabel("Total Population")
plt.grid(True)
plt.show()

def calculate_stability(equilibrium, surfaceArea=36):
    n = len(equilibrium)
    b = [0, 0, 2.89, 10.03, 21.59, 39.02, 56.41, 77.72, 103.23, 131.87, 164.57, 201.46, 242.65]
    s = [0.889, 0.633, 0.697, 0.517, 0.437, 0.287, 0.571, 0.333, 0.75, 1, 0.333, 1]
    
    # Calculate current density and larval survival derivatives
    pop_density = (sum(equilibrium) - equilibrium[0]) / surfaceArea
    exp_term = 520 * np.exp(-0.14 * pop_density)
    denominator = pop_density + exp_term
    L = 2.94 / denominator
    
    # Derivative of L with respect to density (dL/d_rho)
    dL_drho = -2.94 * (1 - 0.14 * exp_term) / (denominator**2)
    
    # Build Jacobian Matrix
    J = np.zeros((n, n))
    
    # Row 1: Partial derivatives of the recruitment function
    total_repro = sum(equilibrium[i] * b[i] for i in range(n))
    for j in range(n):
        # d_rho/d_xj is 1/S for all adult classes (j > 0), and 0 for larvae (j=0)
        drho_dxj = (1.0 / surfaceArea) if j > 0 else 0
        J[0, j] = L * b[j] + total_repro * dL_drho * drho_dxj
    
    # Rows 2-13: Survival transitions
    for i in range(1, n):
        J[i, i-1] = s[i-1]
        
    # Calculate Eigenvalues
    eigenvalues = eigvals(J)
    max_eigen = max(abs(eigenvalues))
    
    return J, max_eigen

# --- Run Analysis ---
# (Assuming 'precise_eq' was found in the previous step)
# precise_eq = [0] * 13
J, stability_index = calculate_stability(precise_eq)

print(f"Dominant Eigenvalue: {stability_index:.4f}")
if stability_index < 1:
    print("The equilibrium is STABLE.")
else:
    print("The equilibrium is UNSTABLE.")