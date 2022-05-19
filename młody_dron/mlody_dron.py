from random import randint, random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def choose_action(state, actions, Q:dict, experiment_rate):
    best_action = None
    best_value = 0

    for action in actions:
        if Q[(state, action)] > best_value:
            best_action = action
            best_value = Q[(state, action)]

    if best_action is None or random() < experiment_rate:
        return actions[randint(0, len(actions) - 1)]
    else:
        return best_action


def mlody_dron():
    # przygotowanie pod wykresy
    episodes = 10000
    moves_until_success = np.array([np.nan for i in range(episodes)])
    fig, axs = plt.subplots(2, 1)

    # Plansza do ruchu - prostokąt 20 x 10
    board = np.array([[0 for i in range(15)] for j in range(10)])
    # currents = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    currents = np.array([randint(0, 2) for i in range(len(board[0]))])

    # Ustawiam początkowy stan drona, położenie celu i chmurę
    # initial_state = (randint(0, len(board[0]) - 1), randint(0, len(board) - 1))
    initial_state = (3, 8)
    # target_pos = (randint(0, len(board[0]) - 1), randint(0, len(board) - 1))
    target_pos = (10, 2)
    toxic_cloud = ((7, 2),
                   (7, 3),
                   (8, 2),
                   (8, 3),
                   (9, 2),
                   (9, 3))
    board[target_pos[1]][target_pos[0]] = 10

    # akcje
    actions = (
        (0, 1),
        (0, -1),
        (-1, 0),
        (1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (0, 0)
    )

    # parametry
    experiment_rate = 0.05
    learning_rate = 0.5
    discount_factor = 0.9

    # funkcja q
    Q = {}
    for i in range(len(board[0])):
        for j in range(len(board)):
            for action in actions:
                Q[((i, j), action)] = 0

    # SARSA
    best_path = []
    best_len = float('inf')
    for episode in range(episodes):
        s1 = initial_state
        a1 = choose_action(s1, actions, Q, experiment_rate)
        curr_path = [s1]
        no_of_moves = 0
        while s1 != target_pos:
            s2 = (s1[0] + a1[0], s1[1] + a1[1])

            # Kara za wyjście poza planszę, wybieram nową akcję
            while not (0 <= s2[0] < len(board[0]) and 0 <= s2[1] < len(board)):
                Q[(s1, a1)] = -10
                a1 = choose_action(s1, actions, Q, experiment_rate)
                s2 = (s1[0] + a1[0], s1[1] + a1[1])

            # rozpatruję prądy powietrzne
            if s2[1] - currents[s2[0]] <= 0:
                s2 = (s2[0], 0)
            else:
                s2 = (s2[0], s2[1] - currents[s2[0]])

            if s2 in toxic_cloud:
                Q[(s1, a1)] = -10
                moves_until_success[episode] = np.nan
                break

            a2 = choose_action(s2, actions, Q, experiment_rate)

            Q[(s1, a1)] = learning_rate * (board[s2[1]][s2[0]] + (discount_factor * Q[(s2, a2)]))\
                + (1 - learning_rate) * Q[(s1, a1)]

            s1 = s2
            a1 = a2
            no_of_moves += 1
            curr_path.append(s1)

        else:
            if no_of_moves < best_len:
                best_len = no_of_moves
                best_path = curr_path
            moves_until_success[episode] = no_of_moves

    axs[0].set_yscale("log")
    axs[0].plot(range(len(moves_until_success)), moves_until_success)
    axs[0].set_title("Ruchy do celu")

    axs[1].set_xlim([0, len(board[0])])
    axs[1].set_ylim([0, len(board)])
    x_locator = mpl.ticker.FixedLocator([i for i in range(len(board[0]))])
    y_locator = mpl.ticker.FixedLocator([i for i in range(len(board))])
    axs[1].xaxis.set_major_locator(x_locator)
    axs[1].yaxis.set_major_locator(y_locator)
    axs[1].grid(visible=True)

    x = np.array([step[0] for step in best_path])
    y = np.array([step[1] for step in best_path])
    axs[1].plot(x, -y + len(board), color='green')

    # wiatry
    for i in range(len(currents)):
        axs[1].scatter([i for _ in range(currents[i])], [j+1 for j in range(currents[i])], color='black', marker='$↑$', s=80)
    # chmura
    for i in toxic_cloud:
        axs[1].scatter([i[0] for i in toxic_cloud], [-i[1] + 10 for i in toxic_cloud], color='C8', marker='$☁$', s=100)

    axs[1].scatter(initial_state[0], -initial_state[1] + len(board), marker='8', s=100)
    axs[1].scatter(target_pos[0], -target_pos[1] + len(board), marker='*', s=200)
    axs[1].set_title("Najlepsza ścieżka")
    fig.tight_layout()
    plt.show()
    print(moves_until_success)


if __name__ == "__main__":
    mlody_dron()
