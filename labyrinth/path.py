
class Path:

    def __init__(self, directions: list[int], fitness: float):
        self.directions = directions
        self.fitness = fitness
    
    def __str__(self) -> str:
        string = "=====\n"
        string += str(self.directions) + "\n"
        string += f"({self.fitness})\n=====\n"
        return string
    
    def get_new_pos(self, direction, pos):
        
        # move up
        if direction == 1:
            return (pos[0], pos[1]+1)

        # move down
        if direction == 2:
            return (pos[0], pos[1]-1)

        # move left
        if direction == 3:
            return (pos[0]-1, pos[1])
        # move right
        if direction == 4:
            return (pos[0]+1, pos[1])
        # stay
        if direction == 5:
            return pos
        
        else:
            print(f'Invalid position: {pos}')
            return pos

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