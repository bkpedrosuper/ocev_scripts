import numpy as np
from math import cos, pow

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