from init_functions import *
from alg_functions import get_fitness_from_individual_maximization, get_fitness_from_individual_minimization, f

if __name__ == "__main__":
    dim, pop, props = get_info_from_base('base1')
    initial_population = create_pop_in_range(max_value=props["MAX"],
                                             min_value=props["MIN"],
                                             decimals=props["DECIMALS"],
                                             pop_size=props["POP"])
    initial_population.append(1.8493)
    initial_fitness_maximization = [get_fitness_from_individual_maximization(x) for x in initial_population]
    initial_fitness_minimization = [get_fitness_from_individual_minimization(x) for x in initial_population]
    initial_fitness_function = [f(x) for x in initial_population]
    
    
    print("Maximization:")
    for individual, fitness, function_value in zip(initial_population, initial_fitness_maximization, initial_fitness_function):
        print(f'{individual} -> {fitness} ({function_value})')
    
    print()
    print("minimization:")
    for individual, fitness, function_value in zip(initial_population, initial_fitness_minimization, initial_fitness_function):
        print(f'{individual} -> {fitness} ({function_value})')

    