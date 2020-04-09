import pyspiel
import numpy as np

game = pyspiel.load_game('othello')
state = game.new_initial_state()
print(state)
print("terminal:", state.is_terminal())
print("current player:", state.current_player())
print(state.legal_actions())

while not state.is_terminal():
    moves = state.legal_actions()
    player = state.current_player()
    state.apply_action(np.random.choice(moves))
    print(moves)
    print(state)

print(state.is_terminal())
print(state.rewards())
print("last player:", player)
print(sum(state.observation_tensor(0)[64:128]))
print(sum(state.observation_tensor(0)[128:]))
