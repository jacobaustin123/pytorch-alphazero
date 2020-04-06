import gym
import torch
import torchvision
import random
import numpy as np

class Game:
    def __init__(self, board_size, num_actions):
        self.board_size = board_size
        self.num_actions = num_actions

    def action_size(self):
        return self.num_actions

    def reward(self, board):
        raise NotImplementedError

    def ended(self, board):
        raise NotImplementedError

    def valid_moves(self, board):
        raise NotImplementedError

    def _print_entry(self, board, x, y):
        raise NotImplementedError

    def move(self, board, code):
        raise NotImplementedError

    def reset(self):
        return torch.zeros(self.board_size, dtype=torch.uint8)

    def flip_board(self, board):
        return torch.stack([board[:,:,1], board[:,:,0]], dim=-1)

    def get_symmetries(self, board, actions):
        pass
        
    def to_string(self, board):
        s = type(self).__name__ + "(["

        for row in range(self.board_size[0]):
            if row != 0:
                s += " " * (len(type(self).__name__) + 2)

            s += "["
            for col in range(self.board_size[1]):
                s += self._print_entry(board, row, col) + " "
            
            s += "]"
            if row != self.board_size[0] - 1:
                s += "\n"
        
        s += "])"

        return s

    def display(self, board):
        print(self.to_string(board))

    def play(self, verbose=False):
        board = self.reset()
        self.display(board)
        curr_player = 0

        while True:
            if verbose:
                print(f"player: {curr_player}, over: {self.ended(board)}, reward: {self.reward(board)}")

            move = int(input())
            board = self.move(board, move)

            if curr_player == 0:   
                self.display(board)
            else:
                self.display(self.flip_board(board))

            board = self.flip_board(board)
            curr_player = 1 - curr_player
