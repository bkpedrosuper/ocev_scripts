import random
def get_info_from_base(base: str):
    with open (f'inputs/{base}') as f:
        dim, pop = 0, 0
        props = {}
        for line in f.readlines():
            op, value = line.split("=")

            if op not in props.keys():
                props[op] = int(value)
            
            if op == "DIM":
                dim = int(value)
            if op == "POP":
                pop = int(value)
        
        return dim, pop, props

def create_int_perm_from_base(base: str):
    dim, pop, props = get_info_from_base(base)

    population = []
    for _ in range(pop):
        individual = [x+1 for x in range(dim)]
        random.shuffle(individual)
        population.append(individual)

    return population       

def create_pop_in_range(max_value: float, min_value: float, decimals: int, pop_size: int):
    population = [round(random.uniform(min_value, max_value), decimals) for _ in range(pop_size)]
    return population