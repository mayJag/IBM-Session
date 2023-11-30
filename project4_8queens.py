def is_safe(board, row, col):
    # Check if it's safe to place a queen in the given position
    # Check the row
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Check upper diagonal on left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check lower diagonal on left side
    for i, j in zip(range(row, len(board), 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, col):
    if col == len(board):
        # All queens are placed successfully
        return True

    for i in range(len(board)):
        if is_safe(board, i, col):
            board[i][col] = 1

            # Recur to place rest of the queens
            if solve_n_queens_util(board, col + 1):
                return True

            # If placing queen in the current position doesn't lead to a solution
            # then remove the queen from the current position (backtrack)
            board[i][col] = 0

    return False

def solve_n_queens(n):
    board = [[0] * n for _ in range(n)]

    if not solve_n_queens_util(board, 0):
        print("Solution does not exist")
        return

    print_solution(board)

def print_solution(board):
    for row in board:
        print(" ".join("Q" if cell == 1 else "." for cell in row))

if __name__ == "__main__":
    solve_n_queens(8)
