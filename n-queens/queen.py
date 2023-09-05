


class Queen:
    def __init__(self, index, pos):
        self.index = index  # Index of the queen
        self.pos = pos      # Position of the queen

    def __str__(self):
        return f"Queen {self.index} at position {self.pos}"
    
class Individual:
    def __init__(self, queens, fitness):
        self.queens = queens
        self.fitness = fitness