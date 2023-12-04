from metro_line import Metro
from math import floor
import copy

class Path:

    def __init__(self, probs: list[float], fitness: float):
        self.probs = probs
        self.fitness = fitness
    
    def __str__(self, metro: Metro) -> str:
        dirs = self.decode(metro=metro)
        directions = []
        for dir in dirs:
            directions.append(dir)
            if dir == metro.end:
                break

        distances = metro.distances
        lines = metro.lines

        curr_line = None
        pos = metro.start
        total_distance = 0
        acc = 0
        lines_changed = 0

        for direction in directions:
            total_distance += distances[pos][direction]

            for color, stations in lines.items():
                if pos in stations and direction in stations:
                    if curr_line != color:
                        lines_changed += 1
                        curr_line = color
                        acc += 5
            
            if direction == metro.end:
                break

            pos = direction

        lines_changed-=1
        total_time = (total_distance / metro.speed) * 60 + acc
    
        string = "=====\n"
        string += f'Path from {metro.start} to {metro.end} \n'
        string += f'Path found: {str(directions)}' + "\n"
        string += f'Lines Changed: {lines_changed}' + "\n"
        string += f"Fitness: ({self.fitness})\n"
        string += f"Distance: {total_distance}Km\n"
        string += f"Time Spent: {total_time:.2f}min\n"
        string += "=====\n"
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