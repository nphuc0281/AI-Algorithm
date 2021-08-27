import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_nodes_from(['S', 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'G'])
G.add_weighted_edges_from(
    [('S', 'A', 3), ('S', 'B', 6), ('S', 'C', 2), ('C', 'E', 1), ('E', 'H', 5), ('E', 'F', 6), ('H', 'G', 8),
     ('F', 'G', 5), ('B', 'E', 2),
     ('B', 'G', 9), ('B', 'D', 4), ('D', 'F', 5), ('A', 'D', 3)])


def DFS(initialState, goalTest):
    frontie = []
    explored = []
    frontie.append(initialState)
    while len(frontie) > 0:
        print(frontie)
        state = frontie.pop(len(frontie) - 1)
        explored.append(state)
        if goalTest == state:
            s = []
            for i in explored:
                s.append(i)
            print(s)
            return True
        for neighbor in G.neighbors(state):
            if neighbor not in list(set(frontie + explored)):
                frontie.append(neighbor)
    return False


if __name__ == '__main__':
    a = DFS('S', 'G')
    # Dinh bat dau la S , Ket thuc la G
    print(a)
    nx.draw(G, with_labels=True)
    plt.show()
