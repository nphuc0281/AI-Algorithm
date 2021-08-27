graph = {
  'S' : ['A', 'B', 'C'],
  'A' : ['S', 'D'],
  'B' : ['S', 'G', 'E', 'D'],
  'C' : ['S', 'E'],
  'D' : ['A', 'B', 'F'],
  'E' : ['B', 'C', 'F', 'H'],
  'F' : ['D', 'E', 'G'],
  'H' : ['E', 'G'],
  'G' : ['B', 'F', 'H']
}

def DFS(initialState, goal):
    frontier = [initialState]
    explored = []
    while frontier:
        print(frontier)
        state = frontier.pop() 
        explored.append(state)
        if goal == state:
            # s = []
            # for i in explored:
            #     s.append(i)
            # print(s)
            return True
        for neighbor in graph[state]:
            if neighbor not in (explored and frontier):
                frontier.append(neighbor)
    return False

DFS('S', 'G')