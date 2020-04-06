import gym
import torch
import torchvision
import random
import numpy as np

from game import Game

class TicTacToe(Game):
    def __init__(self, size=3):
        super(TicTacToe, self).__init__(board_size=(size, size, 2), num_actions=size * size)
    
    def winner(self, board):
        return (board.all(0).any(0) | board.all(1).any(0) |
            board[range(self.board_size[0]), range(self.board_size[1])].all(0) |
            board[range(self.board_size[0]), range(self.board_size[1]-1, -1, -1)].all(0))

    def reward(self, board):
        winner = self.winner(board)
        
        if winner[0]:
            return 1
        elif winner[1]:
            return -1
        elif self.valid_moves(board).sum() == 0:
            return 0.1
        else:
            return 0

    def ended(self, board):
        return bool(self.winner(board).any().item()) or board.any(2).all()

    def flip_board(self, board):
        return torch.stack([board[:,:,1], board[:,:,0]], dim=-1)

    def _print_entry(self, board, x, y):
        if board[x, y, 0]:
            return "x"
        elif board[x, y, 1]:
            return "o"
        else:
            return "-"

    def valid_moves(self, board):
        return ~board.any(2).view(-1)

    def move(self, board, code):
        if self.ended(board):
            raise ValueError("Cannot move after the game has been finished")

        board = board.clone().view(-1, 2)
        if board[code].any():
            raise ValueError("Cannot place a token in a filled position")

        board[code, 0] = True

        return board.view(*self.board_size)
        
    def move_xy(self, board, x, y):
        if self.ended(board):
            raise ValueError("Cannot move after the game has been finished")

        if board[x, y].any():
            raise ValueError("Cannot place a token in a filled position")

        board[x, y, 0] = True

    def get_symmetries(self, board, actions):
        rotations = []
        actions = actions.reshape(self.board_size[0], self.board_size[1])

        for k in range(0, 4):
            for flip in [True, False]:
                new_board = np.rot90(board, k)
                new_actions = np.rot90(actions, k)

                if flip:
                    new_board = np.fliplr(new_board)
                    new_actions = np.fliplr(new_actions)

                rotations.append((torch.tensor(new_board.copy()), torch.tensor(new_actions.ravel().copy())))

        return rotations

if __name__ == "__main__":
    print("USING XY:")
    game = TicTacToe()
    moves = [
        (0, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (2, 2)
    ]

    states = []
    board = game.reset()
    for move in moves:
        game.move_xy(board, move[0], move[1])
        game.display(board)
        print(game.valid_moves(board))

        if move != (2, 2):
            assert not game.ended(board)
        else:
            assert game.reward(board) == 1

        board = game.flip_board(board)


    print("USING CODES:")
    game = TicTacToe()
    moves = [0, 1, 4, 2, 8]

    states = []
    board = game.reset()
    for code in moves:
        board = game.move(board, code)
        game.display(board)

        if code != 8:
            assert not game.ended(board)
        else:
            assert game.reward(board) == 1

        board = game.flip_board(board)

    actions = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    symmetries = game.get_symmetries(board, actions)

    for board, actions in symmetries:
        game.display(board)
        print(actions)