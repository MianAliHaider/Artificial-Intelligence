class Node:
    
    def __init__(self, state, parent, move, h_cost):
        self.state = state
        self.parent = parent
        self.move = move
        self.h_cost = h_cost

    def generate_children(self, goal_state):
        children = []
        x, y = self.find_blank(self.state)
        moves = [('up', x - 1, y), ('down', x + 1, y), ('left', x, y - 1), ('right', x, y + 1)]
        for move, new_x, new_y in moves:
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_state = [list(row) for row in self.state]
                new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
                new_state = tuple(tuple(row) for row in new_state)
                h_cost = self.calculate_heuristic(new_state, goal_state)
                children.append(Node(new_state, self, move, h_cost))
        return children

    def find_blank(self, state):
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    return i, j

    def calculate_heuristic(self, state, goal_state):
        h = 0
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    num = state[i][j]
                    goal_x, goal_y = self.find_position(goal_state, num)
                    h += abs(i - goal_x) + abs(j - goal_y)
        return h

    def find_position(self, state, num):
        for i in range(3):
            for j in range(3):
                if state[i][j] == num:
                    return i, j


class GreedyBestFirstSearch:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state
        self.visited = set()

    def solve(self):
        start_node = Node(self.start_state, None, None, 0)
        start_node.h_cost = start_node.calculate_heuristic(self.start_state, self.goal_state)
        front = [start_node]
        while front:
            front.sort(key=lambda node: node.h_cost)
            current_node = front.pop(0)
            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)
            self.visited.add(current_node.state)
            for child in current_node.generate_children(self.goal_state):
                if child.state not in self.visited:
                    front.append(child)
        return None

    def trace_solution(self, node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

start = ((1, 2, 3),(4, 0, 5),(6, 7, 8))

goal = ((1, 2, 3),(4, 5, 6),(7, 8, 0))

s = GreedyBestFirstSearch(start, goal)
solution = s.solve()

if solution:
    print("Solution found!")
    for path in solution:
        for row in path:
            print(row)
        print()
else:
    print("No solution found.")