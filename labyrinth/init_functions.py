import random
from path import Path
import pandas as pd

def get_info_from_base(base: str):
    with open (f'inputs/{base}') as f:
        props = {}
        for line in f.readlines():
            op, value = line.split("=")

            if op not in props.keys():
                props[op] = int(value)
        
        return props

def create_initial_population(base: str) -> list[Path]:
    props = get_info_from_base(base)
    pop = props["POP"]
    dim = props["DIM"]

    population = []
    directions = [0, 1, 2, 3, 4]
    for _ in range(pop):
        cromossomes = [random.choice(directions)]
        cromossomes.extend([directions[4] for _ in range(dim-1)])
        individual = Path(directions=cromossomes, fitness=0)
        population.append(individual)

    return population


def insert_values(df: pd.DataFrame, values: list, column: int):
    new_values = values.copy()
    last_value = new_values[-1]
    num_repeats = len(df.index) - len(new_values)
    
    if num_repeats < 0:
        num_repeats = 0
    
    new_values.extend([last_value] * num_repeats)
    
    df[column] = new_values
    
    return df

def expand_df(df: pd.DataFrame, expansion: int):
    index_length = len(df.index)
    if index_length < expansion:
        # Calculate the number of repetitions needed for the last element
        repetitions = expansion - index_length
        # Repeat the last element to match the lengths
        new_index = pd.Index(range(0, index_length + repetitions))
        
        # Reindex the DataFrame with the repeated index
        df = df.reindex(new_index)

    df = df.ffill()
    return df
