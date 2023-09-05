import random
from queen import Queen

def get_info_from_base(base: str):
    with open (f'inputs/{base}') as f:
        props = {}
        for line in f.readlines():
            op, value = line.split("=")

            if op not in props.keys():
                props[op] = int(value)
        
        return props

def create_int_perm_from_base(base: str) -> Queen:
    props = get_info_from_base(base)
    pop = props["POP"]
    dim = props["DIM"]

    population = []
    for _ in range(pop):
        pos_values = [x+1 for x in range(dim)]
        random.shuffle(pos_values)
        individual = [Queen(i+1, pos) for i, pos in enumerate(pos_values)]
        population.append(individual)

    return population