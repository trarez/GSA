### Bateman equation https://en.wikipedia.org/wiki/Bateman_equation

import SALib.test_functions
import numpy as np
import matplotlib.pyplot as plt
# from batemaneq import bateman_parent
import SALib

from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from SALib.test_functions import Ishigami
import numpy as np

# Define the model inputs
problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359]]
}

# Generate samples
param_values = sample(problem, 1024)

# Run model (example)
Y = Ishigami.evaluate(param_values)

# Perform analysis
Si = analyze(problem, Y, print_to_console=True)

# Print the first-order sensitivity indices
print(Si['S1'])

# Define function
def bateman_equation(N0, lambdas, t):
    n = len(lambdas)
    result = 0.0
    for i in range(n):
        product = 1.0
        for j in range(n):
            if j != i:
                product *= (lambdas[j] - lambdas[i])
        result += np.exp(-lambdas[i] * t) / product
    return N0 * np.prod(lambdas[:-1]) * result

# Define function for series with plot
def bateman(N0, lambdas, t):
    n = len(lambdas)
    out = np.zeros((t, len(lambdas)))
    for i in range(t):
        for j in range(n):
            out[i,j] = bateman_equation(N0, lambdas[:(j+1)], (i+1))
    for j in range(n):
        plt.plot(out[:,j], label = j+1)
        plt.xlabel('t')
        plt.ylabel('N_n(t)')
        plt.legend()
    plt.show()

# Run example
bateman(N0 = 1.0, lambdas = [0.1, 0.2, 0.3], t = 100)

# Define function for sensitivity analysis, given n = 3 and t = 1
def model(N0, lambdas, t = 1):
    n = len(lambdas)
    out = [None] * n
    for j in range(n):
        out[j] = bateman_equation(int(N0), lambdas[:(j+1)], int(t))
    return out

# Run example
model(N0 = 1, lambdas = [0.1, 0.2, 0.3], t = 1)

# Check with python library
# bateman_parent([0.1, 0.2, 0.3], 1)

# Definizione dei parametri del problema

problem  = { 'num_vars':  4, # Numero di parametri 
            'names': ['N0', 'lambda_1',  'lambda_2', 'lambda_3'],  # Nomi dei parametri 
            'bounds': [[1, 100], [0, 1], [0, 1], [0, 1]] # Range dei parametri
}

# Numero di campioni da generare

num_samples = 1000

param_values = np.zeros((num_samples, problem['num_vars']))


for i in range(problem['num_vars']):
    lower_bound, upper_bound = problem['bounds'][i]
    param_values[:, i] = np.random.uniform(lower_bound, upper_bound, num_samples)


print(param_values)
param_values[1]

output_values = np.array([bateman_equation(params[0],  [params[1], params[2], params[3]], 100) for params in param_values])

for row in param_values:
    bateman_equation(row[0], row[1:], 100)

# Calculate sensitivity indices
for i in range(len(output_values[0])):
    sobol_indices = SALib.sobol_analyze.analyze(problem, output_values[:,i], print_to_console = False)
    print("First order index for", i+1, ":", sobol_indices['S1'])
    print("Total order index for", i+1, ":", sobol_indices['ST'])

# Plot for the last chain
length_bar = 0.35
names_param = ['N0', 'λ1', 'λ2', 'λ3']
plt.bar(np.arange(len(names_param)) - length_bar/2, sobol_indices['S1'], length_bar, label='S1', color='green')
plt.bar(np.arange(len(names_param)) + length_bar/2, sobol_indices['ST'], length_bar, label='ST', color='blue')
plt.xlabel('Parameters')
plt.ylabel('Index')
plt.ylim(0, 1)
plt.title('Sensitivity indexes')
plt.xticks(np.arange(len(names_param)), names_param)
plt.legend()
plt.show()