import numpy as np
from scipy.integrate import odeint
import multiprocessing as mp

class Lorenz63:
    def __init__(self, initial_state, sigma=10.0, rho=28.0, beta=8/3):
        self.state = np.asarray(initial_state, dtype=float)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _lorenz_eq(self, state, t, drift=None):
        if drift is None:
            drift = [0.0,] * 3
        x, y, z = state
        dxdt = self.sigma * (y - x) + drift[0]
        dydt = x * (self.rho - z) - y + drift[1]
        dzdt = x * y - self.beta * z + drift[2]
        return [dxdt, dydt, dzdt]

    def run(self, num_steps, dt, spinup_steps=0, drift=None):
        t = np.arange(0, num_steps*dt, dt)
        result = odeint(self._lorenz_eq, self.state, t, (drift,))
        return result[spinup_steps:]

class Lorenz63Simulator:
    def __init__(self, sigma=10.0, rho=166.1, beta=8/3, rng=None):
        """

        rho (float): 166.08 to 166.2 is intermittent; 28 is default
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.rng = rng

    def _perturb_and_run(self, initial_state, num_steps, dt, spinup_time,
                            perturbation, return_z=True, drift=None):
        perturbed_state = initial_state + perturbation
        lorenz = Lorenz63(perturbed_state, self.sigma, self.rho, self.beta)
        ts = lorenz.run(num_steps, dt, spinup_time, drift=drift)
        if return_z:
            return ts[:,2]
        else:
            return ts

    def generate_perturbations(self, pert_minmax, size):
        # Consider bimodal, gaussian, uniform, LHS?
        perturbations = self.rng.normal(pert_minmax[0], pert_minmax[1], size)
        return perturbations

    def run_simulations(self, truth_initial_state, num_steps, dt, spinup_time,
                            num_perturbations, perturbation_scale, drift_minmax=None):
        truth = Lorenz63(truth_initial_state, self.sigma, self.rho, self.beta)
        truth_series = truth.run(num_steps, dt, spinup_time)

        perturbations = self.generate_perturbations([-perturbation_scale,perturbation_scale],
                                                    (num_perturbations,3))
        drift_perts = self.generate_perturbations([drift_minmax[0],drift_minmax[1]],(num_perturbations,3))

        with mp.Pool(processes=(mp.cpu_count())-1) as pool:
            perturbed_series_list = pool.starmap(self._perturb_and_run,
                        [(truth_initial_state, num_steps, dt, spinup_time, p, True, dp)
                         for p, dp in zip(perturbations,drift_perts)])

        # Only return w/z values
        return truth_series[:,2], perturbed_series_list
