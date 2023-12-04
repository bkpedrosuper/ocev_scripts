import pandas as pd

class Metro:
    def __init__(self, distances: pd.DataFrame, start: str, end: str) -> None:

        self.distances = distances
        self.lines = {
            'r': ['E13', 'E3', 'E9', 'E11'],
            'g': ['E14', 'E13', 'E4', 'E8', 'E12'],
            'b': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
            'y': ['E10', 'E2', 'E9', 'E8', 'E5', 'E7'],
        }
        self.max_distance = distances.sum().max() * 2
        
        self.start = start
        self.end = end
