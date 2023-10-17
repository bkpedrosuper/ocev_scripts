
class Board:
    def __init__(self, matrix: list[list[int]]) -> None:

        start_id = 2
        exit_id = 3
        start: tuple(int, int) = (0, 0)
        exit: tuple(int, int) = (0, 0)

        self.x_size = len(matrix)
        if len(matrix > 0):
            self.y_size = len(matrix[0])
        else:
            self.y_size = 0
            

        # find init and exit
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):

                # Define start
                if matrix[i][j] == start_id:
                    start = (i, j)

                # Define exit
                if matrix[i][j] == exit_id:
                    exit = (i, j)

        self.matrix = matrix
        self.exit = exit
        self.start = start
        print(self.matrix)
        print(self.start)
        print(self.exit)
    
    def get_value(self, pos):
        return self.matrix[pos[0]][pos[1]]