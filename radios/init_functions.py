import random
from radio import Radio

def get_info_from_base(base: str):
    with open (f'inputs/{base}') as f:
        props = {}
        for line in f.readlines():
            op, value = line.split("=")

            if op not in props.keys():
                props[op] = int(value)
        
        return props

def create_initial_population(base: str) -> list[Radio]:
    props = get_info_from_base(base)
    pop = props["POP"]
    dim = props["DIM"]

    population = []
    binar_choice = [0, 1]
    for _ in range(pop):
        cromossomes = [random.choice(binar_choice) for _ in range(dim)]
        individual = Radio(bin=cromossomes, fitness=0)
        population.append(individual)

    return population