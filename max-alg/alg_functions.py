import numpy as np
from math import cos, pow
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def f(x):
    return cos(20*x) - abs(x)/2 + pow(x,3)/4

def get_fitness_from_individual_maximization(x):
    magical_value = 3.25
    answer = magical_value + f(x)
    fitness_value = answer if answer > 0 else 0
    return float(round(fitness_value, 4))

def get_fitness_from_individual_minimization(x):
    magical_value = 2
    answer = magical_value - f(x)
    fitness_value = answer if answer > 0 else 0
    return float(round(fitness_value, 4))

def print_plot_function_with_point(value: float):
    x_vals = np.linspace(-2, 2, 300)
    y_vals = [f(x) for x in x_vals]
    

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot the function line
    plt.plot(x_vals, y_vals, label='Function: f(x) cos(20x) - |x|/2 + x^3/4')

    plt.scatter(value, f(value), color='red')
    plt.scatter(value, f(value), color='red')

    # Annotate the red point with text
    # Annotate the red point with text
    plt.annotate('Red Point', xy=(value, f(value)), xytext=(value + 2, f(value) + 2), arrowprops=dict(facecolor='black', shrink=0.05))

    # Write additional text on the plot
    print(f'Maximization fitness on {value}: {get_fitness_from_individual_maximization(value)}')
    print(f'minimization fitness on {value}: {get_fitness_from_individual_minimization(value)}')

    # Set plot labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Function Plot')

    # Display legend
    plt.legend()



    
    # Show the plot
    plt.grid()
    plt.show()

