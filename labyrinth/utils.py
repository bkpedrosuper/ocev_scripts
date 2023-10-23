from math import sqrt, pow

def euclidean_distance(pos1, pos2):
    x1 = pos1[0]
    y1 = pos1[1]
    x2 = pos2[0]
    y2 = pos2[1]

    distance = sqrt(pow(x2-x1, 2) + pow(y2-y1, 2))
    return distance

def manhattan_distance(pos1, pos2):
    x1 = pos1[0]
    y1 = pos1[1]
    x2 = pos2[0]
    y2 = pos2[1]

    distance = abs(x1 - x2) + abs(y1 - y2)
    return distance
