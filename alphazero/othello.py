import gym
import torch
import torchvision
import random
import numpy as np
from scipy.signal import convolve2d

from alphazero.game import Game

class Othello(Game):
    def __init__(self, size=8):
        super(Othello, self).__init__(board_size=(size, size, 2), num_actions=size * size + 1)

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
        count1, count2 = board.sum((0, 1))

        if not self.ended(board):
            return None

        if count1 > count2: 
            return 0
        elif count2 > count1:
            return 1
        else:
            return None

    def reward(self, board):
        if not self.ended(board):
            return 0

        winner = self.winner(board)
        
        if winner == 0:
            return 1. # win
        elif winner == 1:
            return -1. # loss
        else:
            return 0.1 # tie

    def ended(self, board):
        return not self._valid_moves_no_passing(board).any() and \
            not self._valid_moves_no_passing(self.flip_board(board)).any()

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
        valid_moves = self._valid_moves_no_passing(board)
        
        n_moves = len(valid_moves)
        n_valid_moves = torch.zeros(n_moves + 1)
        n_valid_moves[:n_moves] = valid_moves
        n_valid_moves[n_moves] = valid_moves.sum() == 0 # allow passing only if no valid moves

        return n_valid_moves

    def _valid_moves_no_passing(self, board):
        # return ~board.any(2).view(-1) # simple version
        invalid = board.any(2)
        invalid = invalid | (torch.tensor(convolve2d(board.float()[:,:,1], np.ones((3, 3)), mode='same', boundary='fill')) == 0)

        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                if not invalid[i, j]:
                    for direction in ["up", "down", "left", "right", "up-right", "up-left", "down-right", "down-left"]:
                        steps = self.count_steps(board, i, j, direction)

                        if steps is not None and steps > 0:
                            break
                    else:
                        invalid[i, j] = True

        return ~invalid.view(-1)

    def xy_from_code(self, code):
        if code == self.board_size[0] * self.board_size[1]: # pass
            return None

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

    def _place(self, board, x, y):
        if board[x, y].any():
            raise ValueError("Cannot place a token in a filled position")

        board[x, y, 0] = True

        for direction in ["up", "down", "left", "right", "up-right", "up-left", "down-right", "down-left"]:
            steps = self.count_steps(board, x, y, direction)
            # print(direction, steps)

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
                    self.swap(board, range(x+1, x+1+steps), range(y+1, y+1+steps))
                elif direction == "down-left":
                    self.swap(board, range(x+1, x+1+steps), range(y-steps, y))

        return board

    def move(self, board, code):
        if self.ended(board):
            raise ValueError("Cannot move after the game has been finished")

        move = self.xy_from_code(code)

        if move == None: # pass
            return board.clone()

        x, y = move
        return self._place(board.clone(), x, y)
        
    def move_xy(self, board, x, y):
        if self.ended(board):
            raise ValueError("Cannot move after the game has been finished")

        return self._place(board.clone(), x, y)

    def get_symmetries(self, board, actions):
        rotations = []
        non_pass_actions, p = actions[:-1].reshape(self.board_size[0], self.board_size[1]), actions[-1]

        for k in range(0, 4):
            for flip in [True, False]:
                new_board = np.rot90(board, k)
                new_actions = np.rot90(non_pass_actions, k)

                if flip:
                    new_board = np.fliplr(new_board)
                    new_actions = np.fliplr(new_actions)

                rotations.append((torch.tensor(new_board.copy()), \
                    torch.cat([torch.tensor(new_actions.ravel()), torch.tensor([p])])))

        return rotations
        
if __name__ == "__main__":
    game = Othello(size=4)
    game.play(verbose=True)
    moves = [1, 12, 14, 13, 0, 4, 8, 11, 15, 7]
