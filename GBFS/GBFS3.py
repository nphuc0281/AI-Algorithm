from queue import PriorityQueue

data = {
    'A' : [['B', 3], ['C', 4], ['D', 5]],
    'B' : [['E', 3], ['F', 1]],
    'C' : [['G', 6], ['H', 2]],
    'D' : [['I', 5], ['J', 4]],
    'F' : [['K', 2], ['L', 0], ['M', 4]],
    'H' : [['N', 0], ['O', 4]]
}

class Node:
    def __init__(self, name, parent = None, weight = 6):
        self.name = name
        self.parent = parent
        self.weight = weight

    def display(self):
        print(self.name, self.weight)

    def __lt__(self, other):
        if other == None:
            return False
        return self.weight < other.weight

    def __eq__(self, other):
        if other == None:
            return False
        return self.name == other.name

def equal(O, G):
    if O.name == G.name:
        return True
    return False

def check_priority(temp, c):
    if temp == None:
        return False
    return (temp in c.queue)

def getPath(goal):
    print(goal.name)
    if (goal.parent != None):
        getPath(goal.parent)
    else:
        return

def GBFS(S: Node, G: Node):
    frontier = PriorityQueue()
    explored = PriorityQueue()
    frontier.put(S)
    
    while frontier:
        if frontier == None:
            print("No result")
            return

        state = frontier.get()
        explored.put(state)
        print("At {}({})".format(state.name, state.weight))

        if equal(state, G):
            print("\nResult: ")
            getPath(state)
            return

        if state.name not in data:
            continue
        
        for neigbor in data[state.name]:
            temp = Node(neigbor[0], state, neigbor[1])
            if temp not in (frontier.queue and explored.queue):
                frontier.put(temp)
                temp.display()

if __name__ == '__main__':
    GBFS(Node('A'), Node('N'))