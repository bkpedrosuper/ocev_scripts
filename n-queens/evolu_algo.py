from queen import Queen

def calc_fitness(ind: list[Queen]):

    colisions = 0
    
    for queen1 in ind:
        for queen2 in ind:
            if queen1.index == queen2.index:
                continue

            x = abs(queen1.index - queen2.index)
            y = abs(queen1.pos - queen2.pos)
            
            if x==y:
                colisions+=1

    if colisions == 0:
        return colisions

    return int(colisions / 2)

def generation_manager(population):
    return 2