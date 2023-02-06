import numpy as np
from scipy.signal import lsim2
from scipy.optimize import minimize

def fit_state_space_model(impulse_response, time):
    # Define the state-space model
    def state_space_model(params, input_signal, time):
        A = np.array([[params[0], params[1]], [params[2], params[3]]])
        B = np.array([params[4], params[5]])
        C = np.array([params[6], params[7]])
        D = params[8]
        x0 = np.array([0, 0])

        # Simulate the state-space model
        _, y, _ = lsim2(A, B, C, D, input_signal, time, X0=x0)
        return y
    
    # Define the objective function to minimize the difference between the model response and the impulse response
    def objective_function(params, input_signal, time, target_response):
        model_response = state_space_model(params, input_signal, time)
        difference = model_response - target_response
        return np.sum(difference ** 2)
    
    # Define the initial parameters for the optimization
    params0 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Fit the state-space model to the impulse response
    result = minimize(objective_function, params0, args=(impulse_response, time, impulse_response))
    
    return result.x

# Example input data
impulse_response = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
time = np.arange(-2, 8)

# Fit the state-space model to the impulse response
params = fit_state_space_model(impulse_response, time)

print("Fitted state-space model parameters:", params)
