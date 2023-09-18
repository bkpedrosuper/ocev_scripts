from init_functions import create_initial_population, get_info_from_base
from evolu_algo import generation_manager, bin_to_decimal

if __name__ == "__main__":
    params = get_info_from_base("base1")

    best_values = []
    population = create_initial_population("base1")

    for ind in population:
        print(f'Individual: {ind.bin}')
        print(f'STD {ind.bin[:-5]} -> {bin_to_decimal(ind.bin[:-5])}')
        print(f'LX {ind.bin[-5:]} -> {bin_to_decimal(ind.bin[-5:])}')
        print()

    for gen in range(1, params["GEN"]+1):
        population = generation_manager(population=population, params=params, gen_number=gen)
        best_individual = max(population, key=lambda x: x.fitness)
        best_values.append(best_individual.fitness)