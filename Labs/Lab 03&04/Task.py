import heapq
class PuzzleNode:
    def __init__(self,parent,move,g_cost,h_cost):
        self.parent = parent
        self.move = move
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost  = self.g_cost+self.h_cost
        
    def generate_child(self):
        children = []
        for move in [(-1,0),(1,0),(0,-1),(0,1)]:
            new_state = self.state.copy()  
            if new_state.apply_move(move):
                child = PuzzleNode(new_state, self, move, self.g_cost + 1, self.calculate_heuristic(new_state))
                children.append(child)
        return children
    
    def calculate_heuristic(self,goal_state):
        h = 0
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                if self.state[i][j] != goal_state[i][j]:
                    h += abs(i - goal_state[i][j]) + abs(j - goal_state[j][j])
        return h
                
    
class AStarSolver:
    def __init__(self,start_state,goal_state):
        self.start_state = start_state
        self.goal_state = goal_state
        self.openlist = []
        self.closed = []
        
    def solve(self):
        open = PuzzleNode(self.start_state,None,None,0,0)
        # closed = []
        start_state.h_cost = start_state.calculate_heuristic(self.goal_state)
        # que = []
        heapq.push(self.openlist,self.open)
        while open != []:
            heapq.pop()
            if self.start_state == self.goal_state:
                self.trace_solution()
            # elif     
            
    def trace_solution(self,node):
        solu = []
        while node:
            solu.insert(0,node)
            node = node.parent
        return solu
    def is_solvable(self,state):
        inv = 0
        for i in range(len(state)-1):
            for j in range(i+1,len(state)):
                if i < j:
                    inv += 1
        if len(state) % 2 == 0:
            return inv % 2 == 0
        else:
            return inv % 2 != 0
        
start_state = [
    [1, 2, 3],
    [4, 0, 5],
    [7, 8, 6]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

s = AStarSolver(start_state,goal_state)
if s.is_solvable(start_state):
    print("It is Possbile")
else:
    print("No Possible")