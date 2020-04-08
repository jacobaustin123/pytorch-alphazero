import gym
import torch
import torchvision
import random
import numpy as np

import pyspiel
from alphazero.game import Game

class Wrapper(Game):
    def __init__(self, game):
        self.game = pyspiel.load_game(game)

    def action_size(self):
        return self.game.num_distinct_actions()

    def reward(self, board):        
        if not board.is_terminal():
            return 0

        reward = board.rewards()

        if reward[0] == 0 and reward[1] == 0: # tie
            return 1e-4

        return reward[0] * (-1) ** (board.current_player() != 1)

    def ended(self, board):
        return board.is_terminal()

    def valid_moves(self, board):
        return torch.tensor(board.legal_actions_mask(), dtype=torch.uint8)

    def _print_entry(self, board, x, y):
        raise NotImplementedError

    def move(self, board, code):
        return board.child(code)

    def reset(self):
        return self.game.new_initial_state()

    def flip_board(self, board):
        return board

    def tensor(self, board):
        obs_tensor = torch.tensor(board.observation_tensor(), dtype=torch.uint8)
        player = board.current_player()

        size = self.action_size() - 1
        side = int(np.sqrt(size))
        if player == 1:
            return torch.stack([obs_tensor[2 * size:], obs_tensor[size : 2 * size]]).permute(1, 0).reshape(side, side, 2).to(torch.uint8)
        else:
            return torch.stack([obs_tensor[size : 2 * size], obs_tensor[2 * size:]]).permute(1, 0).reshape(side, side, 2).to(torch.uint8)

    def get_symmetries(self, board, actions):
        rotations = []
        xshape, yshape = self.game.observation_tensor_shape()[1:3]
        non_pass_actions, p = actions[:-1].reshape(xshape, yshape), actions[-1]
        
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

    def to_string(self, board, player=0):
        if (player == 0 and board.current_player() == 1) or (board.current_player() == 0 and player == 1):
            return board.observation_string().replace('o', 'e').replace('x', 'o').replace('e', 'x')
        else:
            return board.observation_string()


    def display(self, board, player=0):
        print(board)

    def play(self, verbose=False):
        board = self.reset()
        self.display(board)
        curr_player = 0
        xshape, yshape = self.game.observation_tensor_shape()[:2]

        while True:
            if verbose:
                print(f"player: {curr_player}, over: {self.ended(board)}, reward: {self.reward(board)}")
                valid_moves = self.valid_moves(board)
                print("moves:", valid_moves.view(xshape, yshape))

            move = int(input())
            board = self.move(board, move)

            if curr_player == 0:   
                self.display(board)
            else:
                self.display(self.flip_board(board))

            board = self.flip_board(board)
            curr_player = 1 - curr_player

        return rotations

if __name__ == "__main__":
    print("USING XY:")
    game = SpielTicTacToe()
    moves = [
        0,
        1,
        4,
        2,
        8
    ]

    states = []
    board = game.reset()
    for move in moves:
        board = game.move(board, move)
        game.display(board)
        # print(board.observation_tensor()[:9])
        # print(board.observation_tensor()[9:18])
        # print(board.observation_tensor()[18:])
        # breakpoint()
        print(game.tensor(board)[:,:,0])
        print(game.tensor(board)[:,:,1])
        print(game.valid_moves(board))
        print(game.reward(board))

        if move != 8:
            assert not game.ended(board)
        else:
            assert game.reward(board) == 1

        board = game.flip_board(board)


    # print("USING CODES:")
    # game = SpielTicTacToe()
    # moves = [0, 1, 4, 2, 8]

    # states = []
    # board = game.reset()
    # for code in moves:
    #     board = game.move(board, code)
    #     game.display(board)

    #     if code != 8:
    #         assert not game.ended(board)
    #     else:
    #         assert game.reward(board) == 1

    #     board = game.flip_board(board)

    # actions = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    # symmetries = game.get_symmetries(board, actions)

    # for board, actions in symmetries:
    #     game.display(board)
    #     print(actions)
