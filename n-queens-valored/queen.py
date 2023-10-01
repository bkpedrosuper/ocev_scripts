
class Queen:
    def __init__(self, index, pos):
        self.pos: int = pos      # Position of the queen
        self.index: int = index  # Index of the queen

    def __str__(self):
        return f"Queen {self.index} at position {self.pos}"
    
class Individual:
    def __init__(self, queens: list[Queen], fitness: float, colisions: int = 9999):
        self.queens = queens
        self.fitness = fitness
        self.colisions = colisions
    
    def __str__(self) -> str:
        text = "========================="
        text += f"Individual ({self.fitness})|({self.colisions})\n ["
        for queen in self.queens:
            text+= str(queen.pos) + ","
        text += "]\n"
        
        return text