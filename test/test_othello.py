import pytest
import alphazero
from alphazero import othello
from alphazero import tictactoe

game = othello.Othello(size=4)
    
class TestOthello:
    def test_sequence(self):
        board = game.reset()
        moves = [1, 0, 4, 8, 15, 7, 2, 3, 16, 11]

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
        
        assert curr_player == 1
        board = game.move(board, moves[-1])
        reward = game.reward(board)
        assert game.ended(board)
        assert reward == 1

    def test_sequence_2(self):
        board = game.reset()
        moves = [14, 15, 11, 13, 0, 3, 7, 1, 2, 16, 12, 4, 8]

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
        
        assert curr_player == 0
        board = game.move(board, moves[-1])
        game.display(board, player=curr_player)
        reward = game.reward(board)
        assert game.ended(board)
        assert reward == 1

    def test_valid(self):
        board = game.reset()
        valid_moves = game.valid_moves(board)

        for i, valid in enumerate(list(valid_moves)):
            if valid:
                n_board = game.move(board, i)
