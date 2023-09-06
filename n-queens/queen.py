

class Queen:
    def __init__(self, index, pos):
        self.index: int = index  # Index of the queen
        self.pos: int = pos      # Position of the queen

    def __str__(self):
        return f"Queen {self.index} at position {self.pos}"
    
class Individual:
    def __init__(self, queens: list[Queen], fitness: float):
        self.queens = queens
        self.fitness = fitness
    
    def __str__(self) -> str:
        stringue = "========================="
        stringue += f"Individual ({self.fitness})\n"
        for queen in self.queens:
            stringue+= (str(queen)) + "\n"
        
        stringue += "========================="
        return stringue