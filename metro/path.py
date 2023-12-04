from metro_line import Metro
from math import floor
import copy

class Path:

    def __init__(self, probs: list[float], fitness: float):
        self.probs = probs
        self.fitness = fitness
    
    def __str__(self) -> str:
        string = "=====\n"
        string += str(self.directions) + "\n"
        string += f"({self.fitness})\n=====\n"
        return string

    def decode(self, metro: Metro) -> list[int]:
        metro_decode = copy.deepcopy(metro)
        start = metro_decode.start
        end = metro_decode.end
        
        directions = []
        
        pos = start
        for prob in self.probs:
            possible_directions = metro_decode.distances[metro_decode.distances[pos].notna()].index
            tp = len(possible_directions)

            new_direction = possible_directions[floor(prob * tp)]
            pos = new_direction
            directions.append(new_direction)
        
        return directions