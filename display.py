import pygame
import numpy as np
import random
import torch
import torch.nn as nn

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
# Inicjalizacja środowiska
class TicTacToeEnv:
    def __init__(self):
        self.model_path = "player3.pth"
        self.loaded_model = DQN2()
        self.loaded_model.load_state_dict(torch.load(self.model_path))
        self.loaded_model.eval()
        self.reset()
    
    def reset(self):
        # Plansza jest reprezentowana jako 1D tablica o długości 9
        # 0: pole puste, 1: znak gracza (X), -1: znak przeciwnika (O)
        self.board = np.zeros(9, dtype=int)
        self.done = False
        self.winner = None
        self.current_player = random.choice([1, -1])
        if self.current_player == -1:
            self.step(self.get_opponent_action())
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

def visualize_tic_tac_toe(env):
    pygame.init()
    
    # Definicje kolorów
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)

    # Wymiary ekranu
    WIDTH, HEIGHT = 300, 500
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Tic Tac Toe')
    
    # Wymiary siatki
    CELL_SIZE = WIDTH // 3
    
    # Font do przycisku resetu i licznika
    FONT = pygame.font.Font(None, 36)
    RESET_BUTTON_RECT = pygame.Rect(50, 310, 200, 30)
    
    # Licznik zwycięstw
    player_1_wins = 0
    player_minus_1_wins = 0

    # Główna pętla gry
    running = True
    while running:
        SCREEN.fill(WHITE)
        
        # Rysowanie siatki
        for x in range(1, 3):
            pygame.draw.line(SCREEN, BLACK, (x * CELL_SIZE, 0), (x * CELL_SIZE, HEIGHT - 200), 3)
            pygame.draw.line(SCREEN, BLACK, (0, x * CELL_SIZE), (WIDTH, x * CELL_SIZE), 3)
        
        # Rysowanie znaków na planszy
        for i in range(9):
            row, col = divmod(i, 3)
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            if env.board[i] == 1:
                pygame.draw.line(SCREEN, RED, (x + 20, y + 20), (x + CELL_SIZE - 20, y + CELL_SIZE - 20), 5)
                pygame.draw.line(SCREEN, RED, (x + CELL_SIZE - 20, y + 20), (x + 20, y + CELL_SIZE - 20), 5)
            elif env.board[i] == -1:
                pygame.draw.circle(SCREEN, BLUE, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 2 - 20, 5)
        
        # Rysowanie przycisku resetu
        pygame.draw.rect(SCREEN, GREEN, RESET_BUTTON_RECT)
        reset_text = FONT.render('Reset', True, WHITE)
        SCREEN.blit(reset_text, (RESET_BUTTON_RECT.x + 50, RESET_BUTTON_RECT.y + 5))
        
        # Rysowanie licznika zwycięstw
        player_1_text = FONT.render(f'Player 1 Wins: {player_1_wins}', True, BLACK)
        player_minus_1_text = FONT.render(f'Player -1 Wins: {player_minus_1_wins}', True, BLACK)
        SCREEN.blit(player_1_text, (10, 360))
        SCREEN.blit(player_minus_1_text, (10, 400))
        
        # Obsługa zdarzeń
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if RESET_BUTTON_RECT.collidepoint(mouse_x, mouse_y):
                    env.reset()
                elif not env.done:
                    col = mouse_x // CELL_SIZE
                    row = mouse_y // CELL_SIZE
                    action = row * 3 + col
                    _, _, done, _ = env.step(action)
                    if done:
                        if env.winner == 1:
                            player_1_wins += 1
                        elif env.winner == -1:
                            player_minus_1_wins += 1
                        env.reset()
                
        pygame.display.flip()
    
    pygame.quit()

# Przykładowe uruchomienie środowiska i wizualizacji
env = TicTacToeEnv()
visualize_tic_tac_toe(env)

