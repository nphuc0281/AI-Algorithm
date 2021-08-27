graph = {
  'A' : ['B','C'],
  'B' : ['A','D','E'],
  'C' : ['A','D','E'],
  'D' : ['B','C','E','F'],
  'E' : ['B','C','D','F'],
  'F' : ['D','E']
}

def BFS(initialState, goal):
    frontier = [initialState]
    explored = []
    while frontier:
        state = frontier.pop(0) 
        explored.append(state)
        if goal == state:
            s = []
            for i in explored:
                s.append(i)
            print(s)
            return True
        for neighbor in graph[state]:
            if neighbor not in (explored and frontier):
                frontier.append(neighbor)
    return False

BFS('A', 'F')