import gym
import torch
import torchvision
import random
import numpy as np

from game import Game

class Othello(Game):
    def __init__(self, size=8):
        super(Othello, self).__init__(board_size=(size, size, 2), num_actions=size * size)

        self.table = {
            "up" : [-1, 0],
            "down" : [1, 0],
            "left" : [0, -1],
            "right" : [0, 1],
            "up-right" : [-1, 1],
            "up-left" : [-1, -1],
            "down-right" : [1, 1],
            "down-left" : [1, -1],
        }

    def reset(self):
        board = torch.zeros(self.board_size, dtype=torch.uint8)
        
        board[self.board_size[0] // 2, self.board_size[1] // 2 - 1, 0] = True
        board[self.board_size[0] // 2 - 1, self.board_size[1] // 2, 0] = True

        board[self.board_size[0] // 2, self.board_size[1] // 2, 1] = True
        board[self.board_size[0] // 2 - 1, self.board_size[1] // 2 - 1, 1] = True

        return board

    def winner(self, board):
        counts = board.sum((0, 1))
        return (counts < counts[0]) & board.any(2).all()

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
        return bool(self.winner(board).any().item()) or bool(board.any(2).all())

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

    def xy_from_code(self, code):
        return (code // self.board_size[0], code % self.board_size[1])

    def swap(self, board, xrange, yrange):
        if isinstance(xrange, int) and isinstance(yrange, int):
            board[xrange, yrange, 1] = False
            board[xrange, yrange, 0] = True
        elif isinstance(xrange, (list, range)):
            board[xrange, [yrange] * len(xrange), 1] = False
            board[xrange, [yrange] * len(xrange), 0] = True
        elif isinstance(yrange, (list, range)):
            board[[xrange] * len(yrange), yrange, 1] = False
            board[[xrange] * len(yrange), yrange, 0] = True

    def on_board(self, x, y):
        return x >= 0 and y >= 0 and x < self.board_size[0] and y < self.board_size[1]

    def get_next(self, x, y, direction):
        vec = np.array(self.table[direction])
        return np.array([x, y]) + vec

    def count_steps(self, board, x, y, direction):
        x, y = self.get_next(x, y, direction)
        count = 0
        while self.on_board(x, y):
            if board[x, y, 1] == False and board[x, y, 0] == True:
                return count
            elif board[x, y, 1] == False and board[x, y, 0] == False:
                return None
            
            count += 1
            x, y = self.get_next(x, y, direction)

        return None

    def place(self, board, x, y):
        if board[x, y].any():
            raise ValueError("Cannot place a token in a filled position")

        board[x, y, 0] = True

        for direction in ["up", "down", "left", "right", "up-right", "up-left", "down-right", "down-left"]:
            steps = self.count_steps(board, x, y, direction)
            
            if steps is not None:
                if direction == "up":
                    self.swap(board, range(x-steps, x), y)
                elif direction == "down":
                    self.swap(board, range(x+1, x+1+steps), y)
                elif direction == "left":
                    self.swap(board, x, range(y-steps, y))
                elif direction == "right":
                    self.swap(board, x, range(y+1, y+steps+1))
                elif direction == "up-right":
                    self.swap(board, range(x-steps, x), range(y+1, y+1+steps))
                elif direction == "up-left":
                    self.swap(board, range(x-steps, x), range(y-steps, y))
                elif direction == "down-right":
                    self.swap(board, range(x, x+steps), range(y+1, y+1+steps))
                elif direction == "down-left":
                    self.swap(board, range(x, x+steps), range(y-steps, y))

        return board

    def move(self, board, code):
        if self.ended(board):
            raise ValueError("Cannot move after the game has been finished")

        x, y = self.xy_from_code(code)
        return self.place(board.clone(), x, y)
        
    def move_xy(self, board, x, y):
        if self.ended(board):
            raise ValueError("Cannot move after the game has been finished")

        return self.place(board.clone(), x, y)

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
    game = Othello(size=4)
    game.play(verbose=True)