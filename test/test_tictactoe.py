import pytest
import torch
import alphazero
from alphazero import openspiel
from alphazero import tictactoe

@pytest.fixture(params=[tictactoe.TicTacToe(3), openspiel.Wrapper('tic_tac_toe')])
def game(request):
    return request.param

class TestTicTacToe:
    def test_sequence(self, game):
        board = game.reset()
        moves = [0, 1, 2, 3, 4, 5, 6]

        curr_player = 0
        for move in moves[:-1]:
            valid_moves = game.valid_moves(board)
            assert valid_moves[move]
            board = game.move(board, move)
            assert not game.ended(board)
            assert game.reward(board) == 0

            if curr_player == 0:
                game.display(board)
            else:
                game.display(game.flip_board(board))
            
            board = game.flip_board(board)
            curr_player = 1 - curr_player
        
        assert curr_player == 0
        board = game.move(board, moves[-1])
        reward = game.reward(board)
        assert game.ended(board)
        assert reward == 1

    def test_sequence_2(self, game):
        board = game.reset()
        moves = [1, 4, 0, 2, 8, 6] # [[x, x, o], [-,o,-], [o, -, x]

        curr_player = 0
        for move in moves[:-1]:
            valid_moves = game.valid_moves(board)
            assert valid_moves[move]
            board = game.move(board, move)
            assert not game.ended(board)
            assert game.reward(board) == 0

            game.display(board, player=curr_player)
            
            board = game.flip_board(board)
            curr_player = 1 - curr_player
        
        assert curr_player == 1
        board = game.move(board, moves[-1])
        game.display(board, player=curr_player)
        reward = game.reward(board)
        expected =  torch.tensor([[[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[1, 1, 0], [0, 0, 0], [0, 0, 1]]], dtype=torch.uint8).permute(1, 2, 0)
        print(expected[:,:,0], expected[:,:,1], game.tensor(board)[:,:,0], game.tensor(board)[:,:,1])
        assert (game.tensor(board) == expected).all()
        assert game.ended(board)
        assert reward == 1

    def test_valid(self, game):
        board = game.reset()
        valid_moves = game.valid_moves(board)

        for i, valid in enumerate(list(valid_moves)):
            if valid:
                n_board = game.move(board, i)
