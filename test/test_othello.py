import pytest
import alphazero
from alphazero import othello
from alphazero import tictactoe
from alphazero import openspiel

# game = othello.Othello(4)
# class TestOthello4:
#     def test_sequence(self):
#         board = game.reset()
#         moves = [1, 0, 4, 8, 15, 7, 2, 3, 16, 11]

#         curr_player = 0
#         for move in moves[:-1]:
#             valid_moves = game.valid_moves(board)
#             assert valid_moves[move]
#             board = game.move(board, move)
#             assert not game.ended(board)
#             assert game.reward(board) == 0

#             if curr_player == 0:
#                 game.display(board)
#             else:
#                 game.display(game.flip_board(board))
            
#             board = game.flip_board(board)
#             curr_player = 1 - curr_player
        
#         assert curr_player == 1
#         board = game.move(board, moves[-1])
#         reward = game.reward(board)
#         assert game.ended(board)
#         assert reward == 1

#     def test_sequence_2(self):
#         board = game.reset()
#         moves = [14, 15, 11, 13, 0, 3, 7, 1, 2, 16, 12, 4, 8]

#         curr_player = 0
#         for move in moves[:-1]:
#             valid_moves = game.valid_moves(board)
#             assert valid_moves[move]
#             board = game.move(board, move)
#             assert not game.ended(board)
#             assert game.reward(board) == 0

#             game.display(board, player=curr_player)
            
#             board = game.flip_board(board)
#             curr_player = 1 - curr_player
        
#         assert curr_player == 0
#         board = game.move(board, moves[-1])
#         game.display(board, player=curr_player)
#         reward = game.reward(board)
#         assert game.ended(board)
#         assert reward == 1

#     def test_valid(self):
#         board = game.reset()
#         valid_moves = game.valid_moves(board)

#         for i, valid in enumerate(list(valid_moves)):
#             if valid:
#                 n_board = game.move(board, i)



@pytest.fixture(params=[othello.Othello(8), openspiel.Wrapper('othello')])
def game(request):
    return request.param
    
class TestOthello8:
    def test_sequence(self, game):
        board = game.reset()
        moves = [26, 20, 45, 42, 12, 17, 43, 44, 41, 19, 18, 10, 
        16, 5, 3, 37, 34, 33, 4, 8, 40, 1, 0, 52, 38, 25, 6, 49, 
        61, 51, 53, 24, 56, 11, 32, 21, 48, 9, 22, 50, 57, 14, 23, 
        60, 29, 7, 59, 54, 13, 2, 47, 30, 31, 63, 15, 62, 46, 39, 55, 58]

        curr_player = 0
        for move in moves[:-1]:
            valid_moves = game.valid_moves(board)
            print(move, curr_player)

            assert valid_moves[move]
            board = game.move(board, move)
            assert not game.ended(board)
            assert game.reward(board, curr_player) == 0

            if curr_player == 0:
                print("--------")
                game.display(board)
            else:
                print("--------")
                game.display(game.flip_board(board))
            
            board = game.flip_board(board)
            curr_player = 1 - curr_player
        
        assert curr_player == 1
        board = game.move(board, moves[-1])
        board = game.flip_board(board)
        curr_player = 1 - curr_player
        assert game.ended(board)

        reward = game.reward(board, player=0) # 1.0, [33, 31]
        assert reward == 1

    def test_sequence_2(self, game):
        board = game.reset()
        moves = [37, 45, 44, 29, 21, 43, 30, 22, 51, 
            59, 34, 23, 54, 19, 18, 17, 9, 20, 50, 26, 38, 
            46, 11, 49, 33, 0, 15, 42, 53, 2, 16, 32, 13, 
            8, 56, 25, 10, 24, 55, 52, 12, 5, 47, 14, 4, 
            3, 60, 41, 57, 62, 1, 61, 7, 48, 63, 40, 31, 39, 6, 64, 58]

        curr_player = 0
        for move in moves[:-1]:
            game.display(board, player=curr_player)
            print(f"MOVE: {move}")

            valid_moves = game.valid_moves(board)
            assert valid_moves[move]

            board = game.move(board, move)
            assert not game.ended(board)
            assert game.reward(board, curr_player) == 0

            print("--------")
            # game.display(board, player=curr_player)
            
            board = game.flip_board(board)
            curr_player = 1 - curr_player
        
        assert curr_player == 0
        board = game.move(board, moves[-1])
        print("--------")
        game.display(board, player=curr_player)
        reward = game.reward(board, 0)
        assert game.ended(board)
        assert reward == 1

    def test_action_size(self, game):
        assert game.action_size() == 65

    def test_valid(self, game):
        board = game.reset()
        valid_moves = game.valid_moves(board)

        for i, valid in enumerate(list(valid_moves)):
            if valid:
                n_board = game.move(board, i)
