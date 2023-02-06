import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lsim2
from scipy.optimize import minimize

def fit_state_space_model(impulse_response, time, n):
    # Define the state-space model
    def state_space_model(params, input_signal, time):
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = params[i * n + j]
        B = params[n ** 2 : n ** 2 + n].reshape(n, 1)
        C = params[n ** 2 + n : n ** 2 + n + n].reshape(1, n)
        D = params[-1]
        x0 = np.zeros((n, 1))

        # Simulate the state-space model
        _, y, _ = lsim2(A, B, C, D, input_signal, time, X0=x0)
        return y
    
    # Define the objective function to minimize the difference between the model response and the impulse response
    def objective_function(params, input_signal, time, target_response):
        model_response = state_space_model(params, input_signal, time)
        difference = model_response - target_response
        return np.sum(difference ** 2)
    
    # Define the initial parameters for the optimization
    params0 = np.ones(n ** 2 + 2 * n + 1)

    # Fit the state-space model to the impulse response
    result = minimize(objective_function, params0, args=(impulse_response, time, impulse_response))
    
    # Extract the state-space matrices from the optimized parameters
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = result.x[i * n + j]
    B = result.x[n ** 2 : n ** 2 + n].reshape(n, 1)
    C = result.x[n ** 2 + n : n ** 2 + n + n].reshape(1, n)
    D = result.x[-1]
    
    return A, B, C, D

# Example input data
impulse_response = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
time = np.arange(-2, 8)
n = 2

# Fit the state-space model to the impulse response
A, B, C, D = fit_state_space_model(impulse_response, time, n)

# Simulate the state-space model with an impulse input
input_signal = np.zeros(len(time))
input_signal[0] = 1
_, model_response, _ = lsim2(A, B, C, D, input_signal, time)

# Plot the original
plt.plot
