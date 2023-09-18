from init_functions import create_initial_population, get_info_from_base
from evolu_algo import generation_manager
from plot_functions import plot_convergence

if __name__ == "__main__":
    params = get_info_from_base("base1")

    best_values = []
    population = create_initial_population("base1")

    # for id, ind in enumerate(population[:-3]):
    #     print(f'Individual {id}:')
    #     print(ind)

    for gen in range(1, params["GEN"]+1):
        population = generation_manager(population=population, params=params, gen_number=gen)
        best_individual = max(population, key=lambda x: x.fitness)
        best_values.append(best_individual.fitness)
        print(f'Best Individual: {best_individual.fitness}')
    
    # for id, ind in enumerate(population[:-3]):
    #     print(f'Individual {id}:')
    #     print(ind)
    
    plot_convergence(generation=params["GEN"], best_values=best_values, n_dim=params["DIM"], save=True)