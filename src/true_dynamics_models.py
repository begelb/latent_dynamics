import math

class LeslieModel3D:
    def __init__(self, th1=19.6, th2=23.68, th3=23.68, survival_p1=0.7, survival_p2=0.7, lower_bounds=[0, 0, 0], upper_bounds=[220, 154, 108]):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.survival_p1 = survival_p1 
        self.survival_p2 = survival_p2 

    def f(self, x):
        return [(self.th1 * x[0] + self.th2 * x[1] + self.th3 * x[2]) * math.exp(-0.1 * (x[0] + x[1] + x[2])), self.survival_p1 * x[0], self.survival_p2 * x[1]]

class LeslieModel4D:
    def __init__(self, th1=80, th2=80, th3=80, th4=80, p1=0.5, p2=0.7, p3=0.7, lower_bounds=[0, 0, 0, 0], upper_bounds=[295, 148, 104, 73]):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.th4 = th4
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.p1 = p1 
        self.p2 = p2 
        self.p3 = p3

    def f(self, x):
        return [(self.th1 * x[0] + self.th2 * x[1] + self.th3 * x[2] + self.th4 * x[3]) * math.exp(-0.1 * (x[0] + x[1] + x[2] + x[3])), self.p1 * x[0], self.p2 * x[1], self.p3 * x[2]]


class RedCoralModel:
    def __init__(self, 
                 b=None, 
                 survival_rates=None, 
                 surface_area=36, 
                 lower_bounds=None, 
                 upper_bounds=None):
        """
        Initialize the Red Coral Population Model.
        
        :param b: List of 13 birth rates (reproductive coefficients).
        :param survival_rates: List of 12 survival probabilities (class i to i+1).
        :param surface_area: Surface area for density calculation (default 36 cm^2).
        """
        # Default birth rates from red_coral.py
        self.b = b if b is not None else [
            0, 0, 2.89, 10.03, 21.59, 39.02, 56.41, 77.72, 103.23, 131.87, 164.57, 201.46, 242.65
        ]
        
        # Default survival rates from red_coral.py
        self.survival_rates = survival_rates if survival_rates is not None else [
            0.889, 0.633, 0.697, 0.517, 0.437, 0.287, 0.571, 0.333, 0.75, 1, 0.333, 1
        ]
        
        self.surface_area = surface_area
        
        # Default bounds based on observed stable equilibrium capacity
        self.lower_bounds = lower_bounds if lower_bounds is not None else [0.0] * 13
     #   self.upper_bounds = upper_bounds if upper_bounds is not None else [1000] * 13
        self.upper_bounds = upper_bounds if upper_bounds is not None else [
            1300, 1150, 750, 520, 270, 120, 35, 20, 7, 5, 5, 2, 2
        ]
        self.dim = 13

    def f(self, x):
        """
        Transition function: calculates the population state at t+1 given state x at t.
        """
        # Calculate adult population density (excluding the first class of recruits)
        # rho = (Total Population - Recruits) / Surface Area
        pop_density = (sum(x) - x[0]) / self.surface_area
        
        # Density-dependent larval survival function L(rho)
        # As density increases, competition reduces the survival of new larvae.
        larval_survival = 2.94 / (pop_density + 520 * math.exp(-0.14 * pop_density))
        
        # Calculate Class 1 (Recruits): 
        # Number of larvae produced by all classes * probability of survival/settlement
        x1_next = larval_survival * sum(x[i] * self.b[i] for i in range(len(x)))
        
        # Calculate Classes 2-13:
        # Simple survival transitions: x_{i+1}(t+1) = x_i(t) * survival_rate_i
        x_rest_next = [x[i] * self.survival_rates[i] for i in range(len(x) - 1)]
       # x_rest_next = [x[i-1] * self.survival_rates[i-1] for i in range(1, self.dim)]
        
        # Combine into the next state vector
        return [x1_next] + x_rest_next