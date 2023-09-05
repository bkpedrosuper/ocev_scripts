from init_functions import create_int_perm_from_base
from evolu_algo import calc_fitness
from queen import Queen

if __name__ == "__main__":
    population = create_int_perm_from_base("base1")
    
    for ind in population:
        [print(queen) for queen in ind]
    print()
    for i, ind in enumerate(population):

        print(f'Fitness for individual {i+1}: {calc_fitness(ind)}')
        print()