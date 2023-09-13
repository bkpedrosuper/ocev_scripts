from init_functions import create_initial_population, get_info_from_base, print_population
from evolu_algo import calc_fitness, generation_manager
from queen import Queen, Individual

if __name__ == "__main__":
    params = get_info_from_base("base1")
    population = create_initial_population("base1")
    
    # print("Initial Population:")
    # print(len(population))
    # print_population(population=population)

    for gen in range(1, params["GEN"]+1):
        population = generation_manager(population=population, params=params, gen_number=gen)
    
    # print("Final Population:")
    # print_population(population=population)