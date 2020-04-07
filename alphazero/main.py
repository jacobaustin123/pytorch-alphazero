import gym
import torch
import numpy as np
import torchvision

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse

import random
import multiprocessing

import datetime
import os

from alphazero.model import TicTacToeNetwork, OthelloNetwork
from alphazero.plotting import VisdomLinePlotter
from alphazero.tictactoe import TicTacToe
from alphazero.othello import Othello
from alphazero.mcts import MCTS

parser = argparse.ArgumentParser(
    description='An implementation of the 2017 DeepMind paper "Mastering the game of \
    Go without human knowledge" for TicTacToe and Othello'
)

NUM_ITER = 100
NUM_EPISODES = 25
BATCH_SIZE = 64
TRAIN_EPOCHS = 10
MAX_TRAIN_SIZE = 20
NUM_MCTS_STEPS = 25
TEMP_THRESHOLD = 15
UPDATE_THRESHOLD = 0.6
SAVE_FREQ = 10
TEST_ALL_FREQ = 25

device = 'cuda:0'

class Arena:
    def __init__(self, player1, player2, game):
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def check_winner(self, board, player):
        reward = self.game.reward(board)
        if reward == 1:
            if player == 0:
                return 1, 0, 0
            else:
                return 0, 0, 1
        elif reward == -1:
            if player == 0:
                return 0, 0, 1
            else:
                return 1, 0, 0
        elif self.game.ended(board):
            return 0, 1, 0
        else:
            return None

    def _play_game(self):
        board = self.game.reset()

        while True:
            action = self.player1(board)
            board = self.game.move(board, action)
            winner = self.check_winner(board, 0)
            if winner is not None:
                return winner

            board = self.game.flip_board(board)
            
            action = self.player2(board)
            board = self.game.move(board, action)
            winner = self.check_winner(board, 1)
            if winner is not None:
                return winner

            board = self.game.flip_board(board)

    def play(self, N=40):
        wins, draws, losses = 0, 0, 0
        for _ in range(int(N / 2)):
            n_wins, n_draws, n_losses = self._play_game()
            wins += n_wins
            draws += n_draws
            losses += n_losses

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(int(N / 2)):
            n_wins, n_draws, n_losses = self._play_game()
            losses += n_wins
            draws += n_draws
            wins += n_losses

        self.player1, self.player2 = self.player2, self.player1

        return wins, draws, losses



class Train:
    def __init__(self, game, network, device='cpu', verbose=False):
        self.device = device
        self.game = game
        self.network = network.to(self.device)
        self.competitor = self.network.clone().to(self.device)

        self.verbose = verbose

        self.mcts = MCTS(self.game, self.network, device=self.device, verbose=self.verbose)

    def episode(self, steps=25, symmetries=True):
        examples = []
        board = self.game.reset()
        curr_player = 0
        episode_step = 0
        
        if self.verbose: self.game.display(board)
        
        while True:
            temp = int(episode_step < TEMP_THRESHOLD)

            pi = self.mcts.get_action_prob(board, temp=temp, steps=steps)

            if self.verbose: print(pi)
            
            if symmetries:
                for b, p in self.game.get_symmetries(board, pi):
                    examples.append((b, p, curr_player))
            else:
                examples.append((board, pi, curr_player))

            action = np.random.choice(len(pi), p=np.array(pi))

            board = self.game.move(board, action)
            if self.verbose: 
                if curr_player == 0:
                    self.game.display(board)
                else:
                    self.game.display(self.game.flip_board(board))
            
            if game.ended(board):
                reward = self.game.reward(board)
                if self.verbose: print(f"Reward: {reward}, Player: {curr_player}")
                return [(board, pi, reward * (-1) ** (curr_player != past_player)) for (board, pi, past_player) in examples]

            board = self.game.flip_board(board)
            curr_player = 1 - curr_player
            episode_step += 1

    def play(self):
        self.network.eval()

        board = self.game.reset()
        self.game.display(board)

        while True:
            action = int(input("move: "))
            board = self.game.move(board, action)
            self.game.display(board)

            reward = self.game.reward(board)

            if reward == 1:
                print(f"You win!")
                return
            elif reward == -1:
                print("The computer wins!")
                return
            elif game.ended(board):
                print("Tie!")
                return

            board = self.game.flip_board(board)
            pi = self.mcts.get_action_prob(board, steps=NUM_MCTS_STEPS)
            _, value = self.network.predict(board.to(self.device))
            print(f"action_probs: {pi}, value: {value}")

            action = pi.argmax()
            print("move: ", action.item())

            board = self.game.move(board, action)

            self.game.display(self.game.flip_board(board))

            reward = self.game.reward(board)

            if reward == 1:
                print(f"The computer win!")
                return
            elif reward == -1:
                print("You win!")
                return
            elif game.ended(board):
                print("Tie!")
                return

            board = self.game.flip_board(board)

    def learn(self):
        self.network.train()
        
        train_examples = []

        for epoch in range(NUM_ITER):
            print(f"epoch {epoch}/{NUM_ITER}")
            
            epoch_examples = []
            for episode in range(NUM_EPISODES):
                self.mcts = MCTS(self.game, self.network, device=self.device, verbose=self.verbose)
                epoch_examples += self.episode(steps=NUM_MCTS_STEPS)

            train_examples.append(epoch_examples)

            if len(train_examples) > MAX_TRAIN_SIZE:
                train_examples.pop(0)
            
            flattened = []
            for e in train_examples:
                flattened.extend(e)
            
            data = torch.stack([episode[0] for episode in flattened])
            policy = torch.stack([episode[1] for episode in flattened])
            values = torch.tensor([episode[2] for episode in flattened])
            
            # print(list(self.network.parameters())[0].data[0])

            self.network.fit(data.to(self.device), [policy.to(self.device), values.to(self.device)], batch_size=BATCH_SIZE, epochs=TRAIN_EPOCHS)

            if epoch == 0:
                self.competitor.load_state_dict(self.network.state_dict())
            
            current_mcts = MCTS(self.game, self.network, device=self.device, verbose=self.verbose)
            competitor_mcts = MCTS(self.game, self.competitor, device=self.device, verbose=self.verbose)

            arena = Arena(
                lambda board : current_mcts.get_action_prob(board, steps=NUM_MCTS_STEPS, temp=0).argmax(),
                lambda board : competitor_mcts.get_action_prob(board, steps=NUM_MCTS_STEPS, temp=0).argmax(),
                self.game
            )

            wins, draws, losses = arena.play(40)
            print(f"New network achieves {wins} wins, {draws} draws, and {losses} losses over the previous iteration.")
            if wins + losses == 0 or wins / (wins + losses) < UPDATE_THRESHOLD: # failed to win enough
                self.network.load_state_dict(self.competitor.state_dict()) # reject, network reverts
            else:
                self.competitor.load_state_dict(self.network.state_dict()) # accept, competitor is current
            
            if epoch % SAVE_FREQ == 0:
                torch.save(self.network.state_dict(), f"backups/network-{epoch}.pt")

            if epoch % TEST_ALL_FREQ == 0:
                self.compare("backups")

    def compare(self, path):
        """
        compare matches the current network against all the saved weights
        in the path directory.
        """
        competitor = self.network.clone().to(self.device)

        for file in os.listdir(path):
            file = os.path.join(path, file)
            if file.endswith(".pt"):
                try:        
                    competitor_state_dict = torch.load(file)
                    competitor.load_state_dict(competitor_state_dict)
                except:
                    print(f"Unable to load state dict for {file}")
                    continue

            current_mcts = MCTS(self.game, self.network, device=self.device, verbose=self.verbose)
            competitor_mcts = MCTS(self.game, competitor, device=self.device, verbose=self.verbose)

            arena = Arena(
                lambda board : current_mcts.get_action_prob(board, steps=NUM_MCTS_STEPS, temp=0).argmax(),
                lambda board : competitor_mcts.get_action_prob(board, steps=NUM_MCTS_STEPS, temp=0).argmax(),
                self.game
            )

            wins, draws, losses = arena.play(20)
            print(f"Current network achieves {wins} wins, {draws} draws, and {losses} losses over {file}")

if __name__ == "__main__":
    # game = TicTacToe(size=3)
    # net = TicTacToeNetwork(3)

    game = Othello(size=4)
    net = OthelloNetwork(size=4)
    # net.load_state_dict(torch.load("backups/othello-4/network-60.pt"))

    train = Train(game, net, device='cuda:0', verbose=False)
    episodes = train.episode(symmetries=False)

    curr_player = 0
    for board, policy, reward in episodes:
        game.display(board, player=curr_player)
        print(f"reward: {reward}, policy: {policy}, player: {curr_player}")
        curr_player = 1 -curr_player

    
    # train.compare("backups")
    # train.learn()
    # train.play()