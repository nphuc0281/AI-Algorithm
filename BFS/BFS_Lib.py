from treelib import Tree,Node

def bfs(tree, s: Node, f):
    frontier = [s]
    explored = []
    while frontier:
        state = frontier.pop(0)
        explored.append(state)
        if f == state.data:
            return explored
        for child in tree.children(state.identifier):
            b1 = is_exist_node(explored, child)
            b2 = is_exist_node(frontier, child)
            if b1 == False and b2 == False:
                frontier.append(child)
    return False


def is_exist_node(list, node):
    for l in list:
        if l.data == node.data:
            return True
    return False


def main():
    tree = Tree()
    tree.create_node("S", "S", data="S")  # root node
    tree.create_node("A", "A", parent="S", data="A")
    tree.create_node("B", "B", parent="S", data="B")
    tree.create_node("C", "C", parent="S", data="C")
    tree.create_node("D", "D1", parent="A", data="D")
    tree.create_node("D", "D2", parent="B", data="D")
    tree.create_node("E", "E1", parent="B", data="E")
    tree.create_node("E", "E2", parent="C", data="E")
    tree.create_node("F", "F1", parent="D1", data="F")
    tree.create_node("F", "F2", parent="E1", data="F")
    tree.create_node("H", "H", parent="E1", data="H")
    tree.create_node("G", "G1", parent="B", data="G")
    tree.create_node("G", "G2", parent="F1", data="G")
    tree.create_node("G", "G3", parent="H", data="G")

    goal = "G"
    result = bfs(tree, tree.get_node("S"), goal)
    if result:
        st = "Order of visit: "
        for r in result:
            st += r.data + " "
        print(st)
    else:
        print("Can not find " + goal)


if __name__ == "__main__":
    main()