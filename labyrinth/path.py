from board import Board
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
    
    def get_new_pos(self, direction, pos):
        up = 0
        down = 1
        left = 2
        right = 3
        stay = 4
        
        # move up
        if direction == up:
            return (pos[0], pos[1]-1)

        # move down
        if direction == down:
            return (pos[0], pos[1]+1)

        # move left
        if direction == left:
            return (pos[0]-1, pos[1])
        # move right
        if direction == right:
            return (pos[0]+1, pos[1])
        # stay
        if direction == stay:
            return pos
        
        else:
            print(f'Invalid position: {pos}')
            return pos

    def decode(self, board: Board) -> list[int]:
        board_decode = copy.deepcopy(board)
        start = board.start

        start_x = board_decode.start[0]
        start_y = board_decode.start[1]
        board_decode.matrix[start_x][start_y] = 0
        
        pos = start

        up = 0
        down = 1
        left = 2
        right = 3
        stay = 4
        directions = []
        

        for prob in self.probs:
            x = pos[0]
            y = pos[1]
            possible_directions = []

            # Mark the current position as a wall
            board_decode.matrix[x][y] = 0

            # Look up
            if not board_decode.get_value(pos=(x, y-1)) == 0:
                possible_directions.append(up)
            
            # Look down
            if not board_decode.get_value(pos=(x, y+1)) == 0:
                possible_directions.append(down)
            
            # Look left
            if not board_decode.get_value(pos=(x-1, y)) == 0:
                possible_directions.append(left)
            
            # Look right
            if not board_decode.get_value(pos=(x+1, y)) == 0:
                possible_directions.append(right)
            
            tp = len(possible_directions)
            if tp == 0:
                new_direction = stay
            else:
                new_direction = possible_directions[floor(prob * tp)]

            # print(f'possible_directions: {possible_directions}')
            # print(f'tp: {tp}')
            # print(f'prob: {prob}')
            # print(f'floor(prob * tp): {floor(prob * tp)}')
            
            directions.append(new_direction)
            pos = self.get_new_pos(new_direction, pos)
        
        return directions

    # def get_info(self) -> str:

    #     directions = ['up', 'down', 'left', 'right', 'stop']
    #     string = "=====\n"
    #     string += str([directions[d] + ',' for d in self.directions]) + "\n"
    #     string += f"({self.fitness})\n=====\n"

    #     max_std = 24
    #     max_lx = 32
    #     max_employees = 40

    #     # CALCULATE SIZE
    #     radio_size = 5
    #     std_bin = self.bin[:-radio_size]
    #     lx_bin = self.bin[-radio_size:]

    #     std = bin_to_decimal(std_bin)
    #     lx = bin_to_decimal(lx_bin)

    #     string += f'Standard: {std}\n'
    #     string += f'Luxury: {lx}\n'

    #     if std > max_std:
    #         string+= f'ALERT: Standard Restriction Violated. Max: {max_std}. Value Found: {std}\n'
        
    #     if lx > max_lx:
    #         string+= f'ALERT: Luxury Restriction Violated. Max: {max_lx}. Value Found: {lx}\n'
        
    #     if std + lx > max_employees:
    #         string+= f'ALERT: Employee Restriction Violated. Max: {max_employees}. Value Found: {std + lx}\n'

    #     string += f'30 * {std} + 40 * {lx}\n'
    #     string += f'${objective_function(std, lx)}\n'
        
    #     return string