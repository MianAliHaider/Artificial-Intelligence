class AlphaBetaPruning: 
    def __init__(self, depth, game_state, player): 
        self.depth = depth
        self.game_state = game_state
        self.player = player
    
    def winner(self, state):
        for row in state:
            if row[0] == row[1] == row[2] and row[0] != " ":
                return row[0]
        for i in range(3):
            if state[0][i] == state[1][i] == state[2][i] and state[0][i] != ' ':
                return state[0][i]
        if state[0][0] == state[1][1] == state[2][2] and state[0][0] != ' ':
            return state[0][0]
        if state[0][2] == state[1][1] == state[2][0] and state[0][2] != ' ':
            return state[0][2]
        return None
    
    def is_terminal(self, state):
        return self.winner(state) is not None or not any(' ' in row for row in state)
    
    def utility(self, state): 
        winner = self.winner(state)
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        else:
            return 0

    def alphabeta(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.is_terminal(state):
            return self.utility(state)
        self.node_count +=1
        if maximizing_player:
            max_val = float('-inf')
            for i in range(3):
                for j in range(3):
                    if state[i][j] == ' ':
                        state[i][j] = 'X'
                        eval = self.alphabeta(state, depth - 1, alpha, beta, False)
                        state[i][j] = ' '
                        max_val = max(max_val, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_val
        else:
            min_val = float('inf')
            for i in range(3):
                for j in range(3):
                    if state[i][j] == ' ':
                        state[i][j] = 'O'
                        eval = self.alphabeta(state, depth - 1, alpha, beta, True)
                        state[i][j] = ' '
                        min_val = min(min_val, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_val

    def best_move(self, state):
        self.node_count = 0 
        best_val = float('-inf')
        best_move = None
        for i in range(3):
            for j in range(3):
                if state[i][j] == ' ':
                    state[i][j] = 'X'
                    move_val = self.alphabeta(state, self.depth, float('-inf'), float('inf'), False)
                    state[i][j] = ' '
                    if move_val > best_val:
                        best_val = move_val
                        best_move = (i, j)
        print(f"Count: {self.node_count}")
        return best_move

def is_draw(state):
    return not any(' ' in row for row in state)

def print_board(state):
    for row in state:
        print(' | '.join(row))
        print('---------')

def play_tic_tac_toe():
    board = [[' '] * 3 for _ in range(3)]
    alphabeta = AlphaBetaPruning(depth=5, game_state=board, player='X')

    while True:
        print_board(board)
        
        move = int(input("Enter your move (0-8): "))  
        row, col = divmod(move, 3)
        if board[row][col] != ' ':
            print("Invalid move, try again.")
            continue
        board[row][col] = 'O'

        if alphabeta.winner(board) == 'O':
            print_board(board)
            print("You win!")
            break
        elif is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        ai_move = alphabeta.best_move(board)
        if ai_move:
            board[ai_move[0]][ai_move[1]] = 'X'
            print("AI chose position", ai_move[0] * 3 + ai_move[1])

        if alphabeta.winner(board) == 'X':
            print_board(board)
            print("AI wins!")
            break
        elif is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

play_tic_tac_toe()
