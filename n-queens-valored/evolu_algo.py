import random
from queen import Queen, Individual
from joblib import Parallel, delayed
import multiprocessing
from board import Board
import numpy as np

def calc_fitness_for_individual(ind: Individual):
    ind.fitness = calc_fitness(ind)


def calc_colisions(ind: Individual):
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

    return colisions


def calc_penalty(ind: Individual, total_colisions: int):
    dimension = len(ind.queens)
    max_colisions = sum([x for x in range(dimension)])
    
    if total_colisions != 0:
        return total_colisions / max_colisions
    
    return 0


def calc_objective_function(ind: Individual, board: Board):
    score = 0
    for queen in ind.queens:
        score += board.matrix[queen.pos -1][queen.index -1]
    return score


def calc_fitness(ind: Individual, board: Board):
    total_colisions = calc_colisions(ind)
    of = calc_objective_function(ind, board)
    penalty = calc_penalty(ind, total_colisions)

    # normalize objective function
    raw_fitness = of / board.max_value
    fitness = raw_fitness - penalty

    return fitness, total_colisions


def routine_tournament_selection(population: list[Individual], fitness_function, tournament_size = 4) -> list[Individual]:
    selected_parents = []
    len_population = len(population)

    kp = 0.6

    if tournament_size > len_population:
        tournament_size = len_population
    
    while len(selected_parents) < len_population:
        choosen_index = 0
        # Randomly select individuals for the tournament
        tournament = random.sample(population, tournament_size)
        
        # Calculate fitness scores for the tournament participants
        fitness_scores = [fitness_function(individual) for individual in tournament]
        
        # Find the index of the individual with the highest fitness score
        best_index = fitness_scores.index(max(fitness_scores))
        worst_index = fitness_scores.index(min(fitness_scores))

        if kp >= random.random():
            choosen_index = best_index
        else:
            choosen_index = worst_index
        
        # Select the best individual from the tournament
        selected_parents.append(tournament[choosen_index])
    
    return selected_parents


def roulette_selection(population: list[Individual], fitness_function, num_parents):
    # Calcula a soma total das aptidões na população
    total_fitness = sum(individual.fitness for individual in population)

    # Gera uma roleta com setores proporcionais às aptidões
    roulette_wheel = []
    for individual in population:
        fitness = individual.fitness
        probability = fitness / total_fitness
        roulette_wheel.extend([individual] * int(probability * 1000))  # Multiplica por 1000 para obter mais precisão

    selected_parents = random.sample(roulette_wheel, num_parents)
    return selected_parents


def crossover_cx(parent1: Individual, parent2: Individual) -> (Individual, Individual):
    ind1 = [queen.pos for queen in parent1.queens]
    ind2 = [queen.pos for queen in parent2.queens]

    first = ind1[0]
    gene1 = ind1[0]
    gene2 = ind2[0]
    size = len(ind1)

    sections_aux = np.array([False for _ in range(size)])
    i = 0
    while gene2 != first:
        i = (i + 1) % size
        gene1 = ind1[i]
        if gene1 == gene2:
            #sections.append(i)
            sections_aux[i] = True
            gene2 = ind2[i]

    mated_ind1 = np.array([ind1[i] if sections_aux[i] is True else ind2[i] for i in range(size)])
    mated_ind2 = np.array([ind2[i] if sections_aux[i] is True else ind1[i] for i in range(size)])

    # print(f'mated_ind1: {mated_ind1}')
    # print(f'mated_ind2: {mated_ind2}')

    offspring1_queens = [Queen(i+1, pos) for i, pos in zip(range(size), mated_ind1)] 
    offspring2_queens = [Queen(i+1, pos) for i, pos in zip(range(size), mated_ind1)]

    offspring1 = Individual(queens=offspring1_queens, fitness=0)
    offspring2 = Individual(queens=offspring2_queens, fitness=0)
    return offspring1, offspring2


def routine_crossover(parents: list[Individual], cross_chance: int) -> list[Individual]:
    parents_size = len(parents)
    offspring = []
    cross_chance = cross_chance / 100

    for i in range(0, parents_size, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        
        if random.random() <= cross_chance:
            (child1, child2) = crossover_cx(parent1, parent2)

            offspring.append(child1)
            offspring.append(child2)
        
        else:
            offspring.append(parent1)
            offspring.append(parent2)
    
    # print(f'PARENT1')
    # print(parent1.__str__())

    # print(f'PARENT2')
    # print(parent2.__str__())

    return offspring


def mutate_individual(individual: Individual, mut_number: int = 1) -> Individual:
    # IMPLEMENT FUNCTION
    new_queens: list[Queen] = individual.queens.copy()
    for _ in range(mut_number):
        pos1, pos2 = random.sample(range(len(new_queens)), 2)
        old_index1 = new_queens[pos1].index
        old_index2 = new_queens[pos2].index

        old_pos1 = new_queens[pos1].pos
        old_pos2 = new_queens[pos2].pos

        new_queens[pos1] = Queen(old_index1, old_pos2)
        new_queens[pos2] = Queen(old_index2, old_pos1)

    new_individual = Individual(queens=new_queens, fitness=0)
    return new_individual


def routine_mutation(population: list[Individual], mut_chance: int) -> list[Individual]:
    mut_chance: float = mut_chance / 100
    for index, individual in enumerate(population):
        if random.random() < mut_chance:
            population[index] = mutate_individual(individual)
    
    return population


def reinsert_elite(current_population: list[Individual], elite_individuals: list[Individual]) -> list[Individual]:
    new_population = current_population.copy()
    for individual in elite_individuals:
        removed_individual: int = random.choice(range(len(current_population) -1))
        new_population[removed_individual] = individual
    
    return new_population


def generation_manager(population: list[Individual], params: dict, gen_number: int, board: Board) -> list[Individual]:
    # print(f'Starting generation {gen_number}')
    individuals_sorted_by_fitness: list[Individual] = sorted(population, key=lambda ind: ind.fitness)
    n_best_individuals: list[Individual] = individuals_sorted_by_fitness[-params["ELIT"]:]

    selected_parents = roulette_selection(population=population, fitness_function=calc_fitness, num_parents=len(population))
    new_population = routine_crossover(selected_parents, cross_chance=params["CROSS"])
    
    new_population = routine_mutation(population, params["MUT"])

    new_population = reinsert_elite(current_population=new_population, elite_individuals=n_best_individuals)

    final_population = []
    for ind in new_population:
        fitness, colisions = calc_fitness(ind, board)
        new_individual = Individual(ind.queens, fitness, colisions)
        final_population.append(new_individual)

    return final_population