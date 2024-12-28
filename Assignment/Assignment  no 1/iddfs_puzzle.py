from copy import deepcopy

def index_empty(puzzle):
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == 0:
                return i, j

def dfs(init_st, goal_st, depth, visi):
    if init_st == goal_st:
        print("Solution Found.")
        return init_st
    if depth <= 0:
        return None
    i, j = index_empty(init_st)
    for move in [(1,0),(-1,0),(0,1),(0,-1)]:
        new_st = deepcopy(init_st)
        new_i,new_j = i + move[0], j + move[1]
        if 0 <= new_i < 3 and 0 <= new_j < 3:   
            new_st[i][j] = new_st[new_i][new_j]
            new_st[new_i][new_j] = 0
            if new_st not in visi:
                visi.append(new_st)
                result = dfs(new_st,goal_st,depth-1,visi)
                if result:
                    return result
    return None

def iddfs(init_st, goal_st, max_depth):
    for depth in range(max_depth):
        visited = [init_st]
        result = dfs(init_st, goal_st, depth, visited)
        if result:
            return result
    return None

def display_puzzle(puzzle):
    for row in puzzle:
        print(row)
    print()

def main():
    init = [[1, 2, 3], [4, 0, 5], [7, 8, 6]]
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    max_depth = 20
    
    result = iddfs(init, goal, max_depth)
    if result:
        display_puzzle(result)
    else:
        print("No Solution exists")
main()
