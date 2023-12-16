import math
class Table:
    def __init__(self, size) -> None:
        self.size = size
        self.map = {}
    def getindex(self, elem):
        if elem not in self.map:
            self.map[elem] = len(self.map)
        return self.map[elem]
def tiles(table, num_tile, state, action):
    disc = []
    for elem in state:
        disc.append(math.floor(elem*num_tile))
    tiles = []
    for i in range(num_tile):
        offset = i
        elements = [i]
        for d in disc:
            elements.append((offset+d)//num_tile)
            offset += i*2
        elements.extend(action)
        tiles.append(table.getindex(tuple(elements)))
    return tiles 
