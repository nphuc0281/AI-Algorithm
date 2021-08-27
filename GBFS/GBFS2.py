from collections import defaultdict
from queue import PriorityQueue

data = defaultdict(list)
data['A'] = ['B', 3, 'C', 4, 'D', 5]
data['B'] = ['E', 3, 'F', 1]
data['C'] = ['G', 6, 'H', 2]
data['D'] = ['I', 5, 'J', 4]
data['F'] = ['K', 2, 'L', 0, 'M', 4]
data['H'] = ['N', 0, 'O', 4]


class Node:
    def __init__(self, name, parent=None, weight=6):
        self.name = name
        self.parent = parent
        self.weight = weight

    def display(self):
        print(self.name, self.weight)

    # overwrite less than
    def __lt__(self, other):
        if other == None:
            return False
        return self.weight < other.weight

    # equal weight
    def __eq__(self, other):
        if other == None:
            return False
        return self.name == other.name

    # bang nhau


def equal(O, G):
    if O.name == G.name:
        return True
    return False


# xet dieu kien :


def check_priority(temp, c):
    if temp == None:
        return False
    return (temp in c.queue)


def getPath(O):
    print(O.name)
    if O.parent != None:
        getPath(O.parent)
    else:
        return


def GreedyBFS(S=Node('A'), G=Node('N')):
    Frontier = PriorityQueue()
    Explored = PriorityQueue()
    Frontier.put(S)
    while True:
        if Frontier == None:
            print("Failed to search")
            return
        O = Frontier.get()
        Explored.put(O)
        print("Duyet : {}({}) ".format(O.name, O.weight))
        if equal(O, G):
            print("Success while searching")
            #  print("Khoang cach: ", O.weight)
            getPath(O)
            return
        i = 0
        
        while i < len(data[O.name]):
            name = data[O.name][i]
            weight = data[O.name][i + 1]
            temp = Node(name=name, weight=weight)
            temp.parent = O
            ok1 = check_priority(temp, Frontier)
            ok2 = check_priority(temp, Explored)
            if not ok1 and not ok2:
                Frontier.put(temp)
                # print("temp: ")
                temp.display();
            i += 2


if __name__ == '__main__':
    GreedyBFS(Node('A'), Node('N'))
