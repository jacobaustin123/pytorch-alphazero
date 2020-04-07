import torch
import numpy as np

from alphazero.model import TicTacToeNetwork, OthelloNetwork
from alphazero.tictactoe import TicTacToe
from alphazero.othello import Othello

CPUCT = 1.0

class MCTS:
    def __init__(self, game, network, device='cuda:0', verbose=False):
        self.P = dict() # array of base probabilities for edge action pair returned by neural net
        self.Q = dict() # value of (s, a) pair
        self.N = dict() # number of times (s, a) pair has been visited
        self.V = dict() # valid actions in state s, basically a cache
        self.terminal = dict() 

        self.game = game
        self.network = network
        self.device = device
        self.verbose = verbose
        
    def get_action_prob(self, state, steps=25, temp=1):
        s = self.game.to_string(state)

        for _ in range(steps):
            self.search(state)
        
        counts = torch.tensor([self.N.get((s, a), 0) for a in range(self.game.action_size())]).to(torch.float)
        
        assert counts.sum() > 0

        if temp == 0:
            best = counts.argmax()
            probs = torch.zeros(self.game.action_size())
            probs[best] = 1
            return probs

        counts = counts ** (1 / temp)
        return counts / counts.sum()

    def search(self, state):
        s = self.game.to_string(state)

        if s in self.terminal:
            value = self.terminal[s]
            return -value

        if s not in self.P:
            if self.game.ended(state):
                self.terminal[s] = self.game.reward(state)
                return -self.terminal[s]

            action_prob, value = self.network(state.unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32).to(self.device))
            action_prob, value = action_prob.detach().cpu(), value.item()

            valid_moves = self.game.valid_moves(state).to(torch.float)

            if valid_moves.sum() == 0: # should always be at least one valid move, even if it's passing
                raise AssertionError("No valid moves found.")

            self.P[s] = valid_moves * action_prob

            if self.P[s].sum() == 0:
                print("[WARNING] zero probability assigned to valid moves.")
                self.P[s] = valid_moves.reshape(1, -1)

            self.P[s] = self.P[s] / self.P[s].sum()
            self.V[s] = valid_moves
            return -value
        else:
            valid_actions = self.V[s]

            qs = torch.tensor([self.Q.get((s, a), 0) for a in range(self.game.action_size())]).to(torch.float)
            ns = torch.tensor([self.N.get((s, a), 0) for a in range(self.game.action_size())]).to(torch.float)
            us = CPUCT * self.P[s] * torch.sqrt(ns.sum() + 1e-8) / (1 + ns)
            dist = us + qs

            assert valid_actions.sum() != 0
            assert (valid_actions.float() * dist != 0).any()

            curr_max = -float("inf")
            action = 0
            for i in range(dist.shape[1]):
                if valid_actions[i] and dist[:,i] > curr_max:
                    curr_max = dist[:,i]
                    action = i

            try:
                n_state = self.game.move(state, action)
            except:
                breakpoint()
                action = np.random.choice(np.where(valid_actions)[0])
                n_state = self.game.move(state, action)

            n_state = self.game.flip_board(n_state)

            v = self.search(n_state)

            if (s, action) in self.Q:
                self.Q[(s, action)] = (self.Q[(s, action)] * self.N[(s, action)] + v) / (self.N[(s, action)] + 1)
                self.N[(s, action)] += 1
            else:
                self.Q[(s, action)] = v
                self.N[(s, action)] = 1

            return -v

if __name__ == "__main__":
    # game = TicTacToe(3)
    # network = TicTacToeNetwork(3)
    
    game = Othello()
    network = OthelloNetwork(8).to('cuda:0')

    board = game.reset()
    mcts = MCTS(game, network, device='cuda:0')

    mcts.search(board)
    print(mcts.get_action_prob(board))
