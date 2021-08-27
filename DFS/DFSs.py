import networkx as nx

def Depth_First_Search(initialState, goalTest):
    frontier = [initialState]
    explored = []
    while len(frontier) > 0:
        state = frontier.pop()
        explored.append(state)
        if goalTest == state:
            return explored
        for neighbor in G.neighbors(state):
            if neighbor not in list(set(frontier + explored)):
                frontier.append(neighbor)
    return False

if __name__ == "__main__":
    G = nx.Graph()
    G.add_nodes_from(["S", "A", "B", "C", "D", "E", "F", "G", "H"])
    G.add_weighted_edges_from(
        [
            ("S", "A", 1),
            ("S", "B", 1),
            ("S", "C", 1),
            ("F", "C", 1),
            ("B", "C", 1),
            ("B", "A", 1),
            ("A", "D", 1),
            ("D", "E", 1),
            ("B", "G", 1),
            ("B", "D", 1),
            ("B", "F", 1),
            ("F", "E", 1),
            ("G", "E", 1),
            ("H", "G", 1),
            ("F", "H", 1),
        ]
    )
    
    result = Depth_First_Search("S", "G")
    if result:
        s = 'Thu tu dinh kham pha: '
        for i in result:
            s += i + " "
        print(s)
    else:
        print('Khong tim thay duong di!')

