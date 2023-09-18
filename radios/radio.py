
class Radio:
    def __init__(self, bin: list[int], fitness: float):
        self.bin = bin
        self.fitness = fitness
    
    def __str__(self) -> str:
        string = "=====\n"
        string += str(self.bin) + "\n"
        string += f"({self.fitness})\n=====\n"
        return string
        
