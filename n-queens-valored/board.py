from math import sqrt, log10

class Board:
    def __init__(self, size) -> None:
        num = 1
        initial_matrix = [[num + i + j for j in range(size)] for i in range(0, size*size, size)]
        transformed_matrix = []
        functions = [sqrt, log10]

        for index, line in enumerate(initial_matrix):
            transformed_matrix.append([functions[index%2](num) for num in line])
        
        self.matrix = transformed_matrix
        self.max_value = sum([line[-1] for line in transformed_matrix])