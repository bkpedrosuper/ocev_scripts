def bin_to_decimal(bin_list: list[int]) -> int:
    decimal = sum([ 2**i if id==1 else 0 for i, id in enumerate(bin_list) ])
    return decimal


def codification(decimal: int, bin_L: int) -> float:
    x_min = 0
    x_max = 24
    x = x_min + ((x_max - x_min) / (2**bin_L -1)) * decimal
    return x


def objective_function(std: float, lx: float):
    return 30 * std + 40 * lx

class Radio:
    def __init__(self, bin: list[int], fitness: float):
        self.bin = bin
        self.fitness = fitness
    
    def __str__(self) -> str:
        string = "=====\n"
        string += str(self.bin) + "\n"
        string += f"({self.fitness})\n=====\n"
        return string
    
    def get_info(self) -> str:
        string = "=====\n"
        string += str(self.bin) + "\n"
        string += f"({self.fitness})\n=====\n"

        max_std = 24
        max_lx = 32
        max_employees = 40

        # CALCULATE SIZE
        radio_size = 5
        std_bin = self.bin[:-radio_size]
        lx_bin = self.bin[-radio_size:]

        std = bin_to_decimal(std_bin)
        lx = bin_to_decimal(lx_bin)

        string += f'Standard: {std}\n'
        string += f'Luxury: {lx}\n'

        if std > max_std:
            string+= f'ALERT: Standard Restriction Violated. Max: {max_std}. Value Found: {std}\n'
        
        if lx > max_lx:
            string+= f'ALERT: Luxury Restriction Violated. Max: {max_lx}. Value Found: {lx}\n'
        
        if std + lx > max_employees:
            string+= f'ALERT: Employee Restriction Violated. Max: {max_employees}. Value Found: {std + lx}\n'

        string += f'30 * {std} + 40 * {lx}\n'
        string += f'${objective_function(std, lx)}\n'
        
        return string