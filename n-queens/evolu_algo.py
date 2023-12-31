import random
from queen import Queen, Individual
from joblib import Parallel, delayed
import multiprocessing

def calc_fitness_for_individual(ind: Individual):
    ind.fitness = calc_fitness(ind)


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


def roulette_selection(population, fitness_function, num_parents):
    # Calcula a soma total das aptidões na população
    total_fitness = sum(fitness_function(individual) for individual in population)

    # Gera uma roleta com setores proporcionais às aptidões
    roulette_wheel = []
    for individual in population:
        fitness = fitness_function(individual)
        probability = fitness / total_fitness
        roulette_wheel.extend([individual] * int(probability * 1000))  # Multiplica por 1000 para obter mais precisão

    selected_parents = random.sample(roulette_wheel, num_parents)
    return selected_parents


def crossover_cx(parent1: Individual, parent2: Individual) -> (Individual, Individual):
    offspring1 = [None] * len(parent1.queens)
    offspring2 = [None] * len(parent2.queens)
    visited = set()

    start = random.randint(0, len(parent1.queens) - 1)

    while True:
        # Get the queen positions at the current position from both parents
        pos_parent1 = parent1.queens[start].pos
        pos_parent2 = parent2.queens[start].pos

        offspring1[start] = parent1.queens[start]
        offspring2[start] = parent2.queens[start]
        visited.add(start)

        position_in_parent1 = [queen.pos for queen in parent1.queens].index(pos_parent2)
        position_in_parent2 = [queen.pos for queen in parent2.queens].index(pos_parent1)

        while position_in_parent1 not in visited:
            pos_parent1 = parent1.queens[position_in_parent1].pos
            pos_parent2 = parent2.queens[position_in_parent1].pos

            offspring1[position_in_parent1] = parent1.queens[position_in_parent1]
            offspring2[position_in_parent1] = parent2.queens[position_in_parent1]
            visited.add(position_in_parent1)

            position_in_parent1 = [queen.pos for queen in parent1.queens].index(pos_parent2)
            position_in_parent2 = [queen.pos for queen in parent2.queens].index(pos_parent1)

        if len(visited) == len(parent1.queens):
            break

        start = random.choice([i for i in range(len(parent1.queens)) if i not in visited])

    offspring1 = Individual(queens=offspring1, fitness=0)
    offspring2 = Individual(queens=offspring2, fitness=0)

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


def generation_manager(population: list[Individual], params: dict, gen_number: int) -> list[Individual]:
    # print(f'Starting generation: {gen_number}')

    individuals_sorted_by_fitness: list[Individual] = sorted(population, key=lambda ind: ind.fitness)
    n_best_individuals: list[Individual] = individuals_sorted_by_fitness[-params["ELIT"]:]

    # selected_parents = roulette_selection(population=population, fitness_function=calc_fitness, num_parents=len(population))
    # new_population = routine_crossover(selected_parents, cross_chance=params["CROSS"])
    
    new_population = routine_mutation(population, params["MUT"])

    new_population = reinsert_elite(current_population=new_population, elite_individuals=n_best_individuals)

    for ind in new_population:
        ind.fitness = calc_fitness(ind)
    # n_jobs = multiprocessing.cpu_count()

    # Parallel(n_jobs=n_jobs)(delayed(calc_fitness_for_individual)(ind) for ind in new_population)
    return new_population