import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

class TicTacToeEnv:
    def __init__(self):
        self.reset()
        self.model_path = "smart_player2.pth"
        self.loaded_model = DQN2()
        self.loaded_model.load_state_dict(torch.load(self.model_path))
        self.loaded_model.eval()
    
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
        
        # Jeśli obecny gracz to przeciwnik, wykonuje ruch
        if self.current_player == -1:
            opponent_action = self.get_opponent_action()
            self.board[opponent_action] = self.current_player
            
            # Sprawdzenie, czy przeciwnik wygrał
            if self.check_win(self.current_player):
                self.done = True
                self.winner = self.current_player
                return self.board.copy(), -1, True, {}
            
            # Sprawdzenie remisu
            if np.all(self.board != 0):
                self.done = True
                self.winner = 0  # Remis
                return self.board.copy(), 0, True, {}
            
            # Powrót do tury gracza
            self.current_player *= -1
        
        return self.board.copy(), 0, False, {}
    
    def get_opponent_action(self):
        # Przeciwnik gra losowo
        # available_actions = np.where(self.board == 0)[0]
        # action = np.random.choice(available_actions)
        # return action
        state_tensor = torch.FloatTensor(self.board).unsqueeze(0)
        q_values = self.loaded_model(state_tensor)
        q_values = q_values.detach().numpy()[0]
        # Maskowanie niedozwolonych akcji
        invalid_actions = np.where(self.board != 0)[0]
        q_values[invalid_actions] = -np.inf
        action = np.argmax(q_values)
        return action
    
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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        # Wybór losowej akcji spośród dostępnych
        available_actions = np.where(state == 0)[0]
        action = np.random.choice(available_actions)
        return action
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state_tensor)
        q_values = q_values.detach().numpy()[0]
        # Maskowanie niedozwolonych akcji
        invalid_actions = np.where(state != 0)[0]
        q_values[invalid_actions] = -np.inf
        action = np.argmax(q_values)
        return action

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

    states = torch.FloatTensor(np.array(batch[0]))
    actions = torch.LongTensor(batch[1]).unsqueeze(1)
    rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
    next_states = torch.FloatTensor(np.array(batch[3]))
    dones = torch.FloatTensor(batch[4]).unsqueeze(1)
    
    # Obliczanie wartości Q dla bieżących stanów
    q_values = policy_net(states).gather(1, actions)
    
    # Obliczanie wartości docelowych
    next_q_values = target_net(next_states).detach()
    max_next_q_values, _ = next_q_values.max(1)
    max_next_q_values = max_next_q_values.unsqueeze(1)
    target_q_values = rewards + (1 - dones) * gamma * max_next_q_values
    
    # Obliczanie straty
    loss = criterion(q_values, target_q_values)
    
    # Aktualizacja modelu
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def draw_table(tab):
    for i in range(0, 9, 3):
        print(f"{tab[i]} | {tab[i+1]} | {tab[i+2]}")
        if i < 6:
            print("_" * 9)

num_episodes = 6000
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.002
learning_rate = 0.001
batch_size = 128
memory_size = 5000
target_update_freq = 50


policy_net = DQN2()
target_net = DQN2()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Ustawienie modelu docelowego w tryb ewaluacji

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

memory = ReplayBuffer(memory_size)

# Inicjalizacja środowiska
env = TicTacToeEnv()
total_reward = 0
total_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        total_reward += reward        
        
        # Zapis doświadczenia
        memory.add((state.copy(), action, reward, next_state.copy(), done))
        state = next_state.copy()

        if 'invalid_move' in info:
            break
        
        # Trening modelu
        optimize_model()
    total_rewards.append(total_reward)
    
    # Aktualizacja epsilon
    if epsilon > epsilon_min:
        epsilon -= epsilon_decay
    
    # Aktualizacja modelu docelowego
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Epizod {episode+1}/{num_episodes}, Aktualna nagroda: {episode_reward,} Łączna nagroda: {total_reward:.1f}, Epsilon: {epsilon:.4f}")

model_path = "smarter_player3.pth"
torch.save(policy_net.state_dict(), model_path)
print(f"Model zapisany do pliku: {model_path}")

average_rewards = total_rewards / np.arange(1, num_episodes + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Total rewards over episodes
ax1.plot(range(101, num_episodes + 1), total_rewards[100:], label="Total Reward")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Total Reward")
ax1.set_title("Total Reward over Episodes")
ax1.legend()
ax1.grid(True)

# Average reward per episode
ax2.plot(range(101, num_episodes + 1), average_rewards[100:], label="Average Reward per Episode")
ax2.set_xlabel("Episodes")
ax2.set_ylabel("Average Reward")
ax2.set_title("Average Reward per Episode over Time")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

