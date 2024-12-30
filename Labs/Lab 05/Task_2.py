class Minimax:
    def __init__(self, game_state):
        self.game_state = game_state

    def is_terminal(self, state):
        return self.winner(state) is not None or not any(' ' in row for row in state)

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
    
    def utility(self, state):
        winner = self.winner(state)
        if winner == 'X':
            return 1
        elif winner == 'O':
            return -1
        else:
            return 0

    def minimax(self, state, depth, maximizing_player):
        if self.is_terminal(state):
            return self.utility(state)
        
        if maximizing_player:
            max_eval = float('-inf')
            for i in range(3):
                for j in range(3):
                    if state[i][j] == ' ':
                        state[i][j] = 'X'
                        eval = self.minimax(state, depth + 1, False)
                        state[i][j] = ' '
                        max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(3):
                for j in range(3):
                    if state[i][j] == ' ':
                        state[i][j] = 'O'
                        eval = self.minimax(state, depth + 1, True)
                        state[i][j] = ' '
                        min_eval = min(min_eval, eval)
            return min_eval

    def best_move(self, state):
        best_move = None
        best_val = float('-inf')
        for i in range(3):
            for j in range(3):
                if state[i][j] == ' ':
                    state[i][j] = 'X'
                    move_val = self.minimax(state, 0, False)
                    state[i][j] = ' '
                    if move_val > best_val:
                        best_val = move_val
                        best_move = (i, j)
        return best_move

def is_draw(state):
    return not any(' ' in row for row in state)

def print_board(state):
    for row in state:
        print(' | '.join(row))
        print('--+---+--+--')

def play_tic_tac_toe():
    board = [[' '] * 3 for _ in range(3)]
    minimax = Minimax(board)

    while True:
        print_board(board)
        
        move = int(input("Enter your move (0-8): "))  
        row, col = divmod(move, 3)
        if board[row][col] != ' ':
            print("Invalid move, try again.")
            continue
        board[row][col] = 'O'

        if minimax.winner(board) == 'O':
            print_board(board)
            print("You win!")
            break
        elif is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        ai_move = minimax.best_move(board)
        if ai_move:
            board[ai_move[0]][ai_move[1]] = 'X'
            print("AI chose position", ai_move[0] * 3 + ai_move[1])

        if minimax.winner(board) == 'X':
            print_board(board)
            print("AI wins!")
            break
        elif is_draw(board):
            print_board(board)
            print("It's a draw!")
            break
play_tic_tac_toe()
