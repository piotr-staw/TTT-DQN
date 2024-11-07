import torch
import numpy as np
import random
from collections import defaultdict
import torch.nn as nn

class TicTacToeEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Plansza jest reprezentowana jako 1D tablica o długości 9
        # 0: pole puste, 1: znak gracza (X), -1: znak przeciwnika (O)
        self.board = np.zeros(9, dtype=int)
        self.done = False
        self.winner = None
        self.current_player = random.choice([1, -1])
        return self.board.copy()
    
    def step(self, action):
        # Sprawdzenie, czy ruch jest dozwolony
        if self.board[action] != 0 or self.done:
            # Nieprawidłowy ruch
            return self.board.copy(), -10, True, {'invalid_move': True}
        
        # Umieszczenie znaku gracza
        self.board[action] = self.current_player
        
        # Sprawdzenie, czy gracz wygrał
        if self.check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            return self.board.copy(), 1, True, {}
        
        # Sprawdzenie remisu
        if np.all(self.board != 0):
            self.done = True
            self.winner = 0  # Remis
            return self.board.copy(), -0.2, True, {}
        
        # Zmiana tury na przeciwnika
        self.current_player *= -1
        
        return self.board.copy(), 0, False, {}
    
    def check_win(self, player):
        # Definicja zwycięskich kombinacji
        winning_combinations = [
            [0,1,2], [3,4,5], [6,7,8],  # Wiersze
            [0,3,6], [1,4,7], [2,5,8],  # Kolumny
            [0,4,8], [2,4,6]            # Przekątne
        ]
        for combo in winning_combinations:
            if np.all(self.board[combo] == player):
                return True
        return False
    
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 9)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
    
class DQN2(nn.Module):
    def __init__(self):
        super(DQN2, self).__init__()
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 9)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        return self.fc5(x)
    
    
def load_model(model_path, model_class):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load models
players = {
    "player1": load_model("player1.pth", DQN),
    "player2": load_model("player2.pth", DQN),
    "player3": load_model("player3.pth", DQN2),
    "player4": load_model("player4.pth", DQN2)
}

# Run tournament
n_games = 100
results = defaultdict(lambda: defaultdict(int))

for p1_name, p1_model in players.items():
    for p2_name, p2_model in players.items():
        if p1_name == p2_name:
            continue
        
        p1_wins, p2_wins, draws = 0, 0, 0
        
        for i in range(n_games):
            env = TicTacToeEnv()
            
            while not env.done:
                if env.current_player == 1:
                    state_tensor = torch.FloatTensor(env.board).unsqueeze(0)
                    q_values = p1_model(state_tensor).detach().numpy()[0]
                else:
                    state_tensor = torch.FloatTensor(env.board).unsqueeze(0)
                    q_values = p2_model(state_tensor).detach().numpy()[0]
                
                # Maskowanie niedozwolonych akcji
                invalid_actions = np.where(env.board != 0)[0]
                q_values[invalid_actions] = -np.inf
                action = np.argmax(q_values)
                
                env.step(action)
            
            # Record the outcome
            if env.winner == 1:
                p1_wins += 1
            elif env.winner == -1:
                p2_wins += 1
            else:
                draws += 1

            print(f"Game {i}/{n_games} - {p1_name}: {p1_wins}, {p2_name}: {p2_wins}, draws: {draws}")
        
        # Store results for each match-up
        results[p1_name][p2_name] = {
            "wins": p1_wins,
            "losses": p2_wins,
            "draws": draws
        }

# Print final statistics
for p1_name in players:
    for p2_name in players:
        if p1_name == p2_name:
            continue
        res = results[p1_name][p2_name]
        print(f"{p1_name} vs {p2_name}: {res['wins']} wins, {res['losses']} losses, {res['draws']} draws")
    total_wins = sum(1 for p2_name in players if p1_name != p2_name and results[p1_name][p2_name]['wins'] > results[p1_name][p2_name]['losses'])
    print(f"{p1_name} won against {total_wins} opponents out of {len(players) - 1}")
    print()
