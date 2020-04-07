import pytest
import alphazero
from alphazero import othello
from alphazero import tictactoe

@pytest.fixture(params=[othello.Othello(size=4), tictactoe.TicTacToe(size=3)])
def game(request):
    return request.param

class TestGames:
    @pytest.mark.xfail()
    def test_place(self, game):
        board = game.reset()
        game.move(board, 0)
        game.move(board, 0)

    def test_valid(self, game):
        board = game.reset()
        valid_moves = game.valid_moves(board)

        for i, valid in enumerate(list(valid_moves)):
            if valid:
                n_board = game.move(board, i)