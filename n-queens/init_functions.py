import random
from queen import Queen, Individual
from evolu_algo import calc_fitness

def get_info_from_base(base: str):
    with open (f'inputs/{base}') as f:
        props = {}
        for line in f.readlines():
            op, value = line.split("=")

            if op not in props.keys():
                props[op] = int(value)
        
        return props

def create_initial_population(base: str) -> list[Individual]:
    props = get_info_from_base(base)
    pop = props["POP"]
    dim = props["DIM"]

    population = []
    for _ in range(pop):
        pos_values = [x+1 for x in range(dim)]
        random.shuffle(pos_values)
        individual_queens: list[Queen] = [Queen(i+1, pos) for i, pos in enumerate(pos_values)]
        individual = Individual(queens=individual_queens, fitness=0)
        population.append(individual)

    return population

def print_population(population: list[Individual]):
    individuals_sorted_by_fitness: list[Individual] = sorted(population, key=lambda ind: ind.fitness)
    for i, ind in enumerate(individuals_sorted_by_fitness):
        print(f'Individual: {i+1}')

        print(ind)