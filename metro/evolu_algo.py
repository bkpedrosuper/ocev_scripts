import random
from path import Path
import numpy as np
from metro_line import Metro
from utils import euclidean_distance, manhattan_distance


def calc_penalty(max_distance: float, found_distance: float):
    return found_distance / max_distance

def calc_fitness(ind: Path, metro: Metro) -> float:
    directions = ind.decode(metro=metro)
    distances = metro.distances
    lines = metro.lines

    curr_line = None
    pos = metro.start
    total_distance = 0
    acc = 0

    for direction in directions:
        total_distance += distances[pos][direction]
        
        for color, stations in lines.items():
            if pos in stations and direction in stations:
                if curr_line != color:
                    curr_line = color
                    acc += 5
        
        if direction == metro.end:
            break
    
    penalty = calc_penalty(metro.max_distance, total_distance)
    
    return penalty

def routine_tournament_selection(population: list[Path], fitness_function, tournament_size = 4) -> list[Path]:
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
        fitness_scores = [individual.fitness for individual in tournament]
        
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


def roulette_selection(population, fitness_function, num_parents) -> list[Path]:
    # Calcula a soma total das aptidões na população
    total_fitness = sum([individual.fitness for individual in population])

    # Gera uma roleta com setores proporcionais às aptidões
    roulette_wheel = []
    for individual in population:
        fitness = individual.fitness
        probability = fitness / total_fitness
        roulette_wheel.extend([individual] * int(probability * 1000))  # Multiplica por 1000 para obter mais precisão

    selected_parents = random.sample(roulette_wheel, num_parents)
    return selected_parents


def one_point_crossover(parent1: Path, parent2: Path) -> (Path, Path):
    cr1 = parent1.probs
    cr2 = parent1.probs
    
    position = random.randint(0, len(cr1) - 1)

    cr1_floor, cr1_ceil = cr1[:position], cr1[position:]
    cr2_floor, cr2_ceil = cr2[:position], cr2[position:]

    mated_cr1 = np.concatenate((cr1_floor, cr2_ceil), axis=0)
    mated_cr2 = np.concatenate((cr2_floor, cr1_ceil), axis=0)

    return Path(mated_cr1, 0), Path(mated_cr2, 0)


def routine_crossover(parents: list[Path], cross_chance: int) -> list[Path]:
    parents_size = len(parents)
    offspring = []
    cross_chance = cross_chance / 100

    for i in range(0, parents_size, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        
        if random.random() <= cross_chance:
            child1, child2 = one_point_crossover(parent1, parent2)

            offspring.append(child1)
            offspring.append(child2)
        
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    return offspring



def mutate_individual(individual: Path, mut_chance: float) -> Path:
    new_probs = []

    for direction in individual.probs:
        if random.random() < mut_chance:
            new_probs.append(random.random())
        else:
            new_probs.append(direction)

    individual = Path(new_probs, 0)
    return individual


def routine_mutation(population: list[Path], mut_chance: int) -> list[Path]:
    mut_chance: float = mut_chance / 100
    for index, individual in enumerate(population):
        population[index] = mutate_individual(individual, mut_chance)
    
    return population


def reinsert_elite(current_population: list[Path], elite_individuals: list[Path]) -> list[Path]:
    new_population = current_population.copy()
    for individual in elite_individuals:
        removed_individual: int = random.choice(range(len(current_population)))
        new_population[removed_individual] = individual
    
    return new_population


def generation_manager(population: list[Path], params: dict, gen_number: int, metro: Metro) -> list[Path]:
    print(f'Starting generation {gen_number}')

    individuals_sorted_by_fitness: list[Path] = sorted(population, key=lambda ind: ind.fitness)
    n_best_individuals: list[Path] = individuals_sorted_by_fitness[-params["ELIT"]:]

    selected_parents = routine_tournament_selection(population=population, fitness_function=calc_fitness, tournament_size=3)
    new_population = routine_crossover(selected_parents, cross_chance=params["CROSS"])
    
    new_population = routine_mutation(population, params["MUT"])

    new_population = reinsert_elite(current_population=new_population, elite_individuals=n_best_individuals)

    final_population = []
    for ind in new_population:
        probs = ind.probs
        fitness = calc_fitness(ind, metro)
        new_individual = Path(probs, fitness)
        final_population.append(new_individual)

    return final_population