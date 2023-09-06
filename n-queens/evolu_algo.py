import random
from queen import Queen, Individual

def calc_fitness(ind: Individual):

    dimension = len(ind.queens)
    min_fitness = 0
    max_fitness = sum([x for x in range(dimension)])

    colisions = 0
    
    for queen1 in ind.queens:
        for queen2 in ind.queens:
            if queen1.index == queen2.index:
                continue

            x = abs(queen1.index - queen2.index)
            y = abs(queen1.pos - queen2.pos)
            
            if x==y:
                colisions+=1

    if colisions != 0:
        colisions = int(colisions / 2)
    
    raw_fitness = max_fitness - colisions

    normalized_fitness = (raw_fitness - min_fitness) / (max_fitness - min_fitness)

    return normalized_fitness

def routine_tournament_selection(population: list[Individual], fitness_function, tournament_size = 4) -> list[Individual]:
    selected = []
    len_population = len(population)

    if tournament_size > len_population:
        tournament_size = len_population
    
    for _ in range(len(population)):
        # Randomly select individuals for the tournament
        tournament = random.sample(population, tournament_size)
        
        # Calculate fitness scores for the tournament participants
        fitness_scores = [fitness_function(individual) for individual in tournament]
        
        # Find the index of the individual with the highest fitness score
        best_index = fitness_scores.index(max(fitness_scores))
        
        # Select the best individual from the tournament
        selected.append(tournament[best_index])
    
    return selected

def crossover_cx(parent1: Individual, parent2: Individual) -> (Individual, Individual):
    # IMPLEMENT FUNCTION
    return parent1, parent2

def routine_crossover(parents: list[Individual], cross_chance: int) -> list[Individual]:
    parents_size = len(parents)
    offspring = []
    cross_chance = cross_chance / 100

    for i in range(0, parents_size, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        
        if random.random() <= cross_chance:
            child1, child2 = crossover_cx(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)

    return offspring


def mutate_individual(individual: Individual):
    # IMPLEMENT FUNCTION
    return individual


def routine_mutation(population: list[Individual], mut_chance: int) -> list[Individual]:
    mut_chance: float = mut_chance / 100
    for index, individual in enumerate(population):
        if random.random() < mut_chance:
            population[index] = mutate_individual(individual)
    
    return population
    

def generation_manager(population: list[Individual], params: dict) -> list[Individual]:
    for ind in population:
        ind.fitness = calc_fitness(ind)

    individuals_sorted_by_fitness = sorted(population, key=lambda ind: ind.fitness)
    n_best_individuals = individuals_sorted_by_fitness[:params["ELIT"]]

    selected_parents = routine_tournament_selection(population=population, fitness_function=calc_fitness)
    new_population = routine_crossover(selected_parents, cross_chance=params["CROSS"])
    new_population = new_population + n_best_individuals
    new_population = new_population[:params["POP"]]

    new_population = routine_mutation(new_population, params["MUT"])

    return population