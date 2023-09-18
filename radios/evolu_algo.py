import random
from radio import Radio


def calc_penalty(ind: Radio):
    std_bin = ind.bin[:-5]
    lx_bin = ind.bin[-5:]
    
    std = 30 * bin_to_decimal(std_bin)
    lx = 40 * bin_to_decimal(lx_bin)

    partial_penalty = (std + 2*lx - 40) / 16
    penalty = max([0, partial_penalty])
    return penalty


def calc_fitness(ind: Radio, bin_L: int) -> float:
    std_bin = ind.bin[:-5]
    lx_bin = ind.bin[-5:]
    
    std = 30 * bin_to_decimal(std_bin)
    lx = 40 * bin_to_decimal(lx_bin)



    profit = std + lx
    c = -1
    penalty = calc_penalty(ind)

    # NORMALIZE FITNESS
    max_profit = 30 * 24 + 40 * 16
    min_profit = 0


    raw_fitness = codification(profit, bin_L) + (-1) * codification(penalty, bin_L)
    normalized_fitness = (raw_fitness - min_profit) / (max_profit - min_profit)


    return normalized_fitness


def bin_to_decimal(bin_list: list[int]) -> int:
    decimal = sum([ 2**i if id==1 else 0 for i, id in enumerate(bin_list) ])
    return decimal


def codification(decimal: int, bin_L: int) -> float:
    x_min = 0
    x_max = 24
    x = x_min + ((x_max - x_min) / (2**bin_L -1)) * decimal
    return x


def routine_tournament_selection(population: list[Radio], fitness_function, tournament_size = 4) -> list[Radio]:
    selected_parents = []
    len_population = len(population)

    kp = 0.9

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


def roulette_selection(population, fitness_function, num_parents) -> list[Radio]:
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


def one_point_crossover(parent1: Radio, parent2: Radio) -> (Radio, Radio):
    return parent1, parent2


def routine_crossover(parents: list[Radio], cross_chance: int) -> list[Radio]:
    parents_size = len(parents)
    offspring = []
    cross_chance = cross_chance / 100

    for i in range(0, parents_size, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        
        if random.random() <= cross_chance:
            (child1, child2) = one_point_crossover(parent1, parent2)

            offspring.append(child1)
            offspring.append(child2)
        
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    return offspring


def mutate_individual(individual: Radio, mut_number: int = 1) -> Radio:
    # IMPLEMENT FUNCTION
    
    return individual


def routine_mutation(population: list[Radio], mut_chance: int) -> list[Radio]:
    mut_chance: float = mut_chance / 100
    for index, individual in enumerate(population):
        if random.random() < mut_chance:
            population[index] = mutate_individual(individual)
    
    return population


def reinsert_elite(current_population: list[Radio], elite_individuals: list[Radio]) -> list[Radio]:
    new_population = current_population.copy()
    for individual in elite_individuals:
        removed_individual: int = random.choice(range(len(current_population) -1))
        new_population[removed_individual] = individual
    
    return new_population


def generation_manager(population: list[Radio], params: dict, gen_number: int) -> list[Radio]:
    
    for ind in population:
        ind.fitness = calc_fitness(ind)

    individuals_sorted_by_fitness: list[Radio] = sorted(population, key=lambda ind: ind.fitness)
    n_best_individuals: list[Radio] = individuals_sorted_by_fitness[-params["ELIT"]:]

    selected_parents = roulette_selection(population=population, fitness_function=calc_fitness, num_parents=len(population))
    new_population = routine_crossover(selected_parents, cross_chance=params["CROSS"])
    
    new_population = routine_mutation(population, params["MUT"])

    new_population = reinsert_elite(current_population=new_population, elite_individuals=n_best_individuals)

    for ind in new_population:
        ind.fitness = calc_fitness(ind)

    return new_population