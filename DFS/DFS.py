class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
    def addChild(self, list):
        for c in list:
            self.children.append(c)

nodeA = Node("A")
nodeB = Node("B")
nodeC = Node("C")
nodeD = Node("D")
nodeE = Node("E")
nodeF = Node("F")
nodeG = Node("G")
nodeA.addChild([nodeB, nodeC, nodeD])
nodeD.addChild([nodeE])
nodeC.addChild([nodeF,nodeG])

def DFS(initialState, goal):
    frontier = [initialState]
    explored = []
    while frontier:
        state = frontier.pop()
        explored.append(state)
        if goal == state.name:
            for i in explored:
                s = i.name + " "
                print(s)
            return True
        for child in state.children:
            if child not in (explored and frontier):
                frontier.append(child)
    return False

DFS(nodeA, "F")