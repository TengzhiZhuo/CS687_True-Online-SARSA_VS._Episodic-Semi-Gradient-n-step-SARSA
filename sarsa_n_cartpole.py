import gym
import tile
import numpy as np
import random
import math
import matplotlib.pyplot as plt 

class GridTiling:
    def __init__(self, size, num_tile) -> None:
        self.iht = tile.Table(size)
        self.num_tile = num_tile
        self.position_scale = self.num_tile / (4.8 + 4.8)
        self.angle_scale = self.num_tile / (0.418 + 0.418)

    def get_tiles(self, position, velocity, angle, d_angle, action):
        return np.array(tile.tiles(self.iht, self.num_tile, [position*self.position_scale, velocity, angle*self.angle_scale, d_angle], [action]))

def q_value(w, tiles):
    return np.dot(w, tiles)

def find_active_tiles(state, action, tile):
    x_sa = tile.get_tiles(state[0], state[1], state[2], state[3], action)
    active_tiles = np.zeros(num_tile**4,)
    for elem in x_sa:
        active_tiles[elem] = 1.0
    return active_tiles

def q(state, tile, w):
    q_values = []
    for action in range(2):
        active_tiles = find_active_tiles(state, action, tile)
        q_values.append(np.dot(w, active_tiles))
    action = np.argmax(q_values)
    return action, q_values[action]

def episodic_Sarsa(episodes, env, alpha, n, gamma):
    w = np.zeros(num_tile**4,)
    tile = GridTiling(num_tile**4, num_tile)
    steps = []
    total_steps = 0
    sum = []

    for _ in range(episodes):
        state = env.reset()[0]
        St, At, Rt = [state], [], [-1.0]
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() 
        else:
            action, value = q(state, tile, w)
        At.append(action)
        T = math.inf
        t = 0
        while True:
            if t < T:
                state, reward, terminated, truncated, info = env.step(At[t])
                Rt.append(reward)
                St.append(state)
                if truncated:
                    break
                if terminated:
                    T = t+1 
                else:
                    if random.uniform(0, 1) < epsilon:
                        action = env.action_space.sample() 
                    else:
                        action, value = q(state, tile, w)
                    At.append(action)
            rho = t-n+1  
            if rho >= 0:
                G = 0.0
                top = min(rho+n, T)
                for i in range(rho+1, top+1):
                    G += gamma**(i-rho-1)*Rt[i]
                if (rho+n) < T:
                    active_tiles = find_active_tiles(St[rho+n], At[rho+n], tile)
                    value = q_value(w, active_tiles)  
                    G = G + gamma**n * value
                active_tiles = find_active_tiles(St[rho], At[rho], tile)
                value = q_value(w, active_tiles)
                w += np.multiply(alpha*(G-value), active_tiles)
            if rho == T-1:
                total_steps += t 
                steps.append(total_steps)
                break
            t = t+1
        sum.append(t)
    return w, tile, np.array(steps), np.array(sum)

def evluation(w, tiling):
    tile = tiling
    steps = []
    for _ in range(200):
        step = 0
        env = gym.make('CartPole-v1')
        state = env.reset()[0]
        for i in range(500):
            action, _ = q(state, tile, w)
            next_state, _, terminated, _, _ = env.step(action)
            state = next_state
            step += 1
            if terminated:
                break
        env.close()
        steps.append(step)
    return np.mean(step)

env = gym.make("CartPole-v1")
alpha = [0.3/8, 0.2/8, 0.5/8]
ns = [4, 6, 8]
gamma = [1.0, 0.9, 0.92]
epsilon = 0.1
num_tile = 8

best_alpha = None
best_n = None
best_gamma = None
best_steps = -math.inf
for a in alpha:
    for n in ns:
        for g in gamma:
            w, tiling, steps, sum = episodic_Sarsa(350, env, a, n, g)
            if np.mean(sum) >= best_steps:
                best_alpha = a
                best_n = n
                best_gamma = g
                best_steps = np.mean(sum)

print(best_alpha, best_n, best_gamma)

best_alpha = 0.0625
best_n = 4
best_gamma = 1.0
epsilon = 0.1
num_tile = 8


from tqdm import tqdm

env = gym.make("CartPole-v1")
y = range(0, 350)
total_steps = np.zeros(350)
average_to_goal = []
for _ in tqdm(range(10)):
    w, tiling, _, steps = episodic_Sarsa(350, env, best_alpha, best_n, best_gamma)
    total_steps += steps
    average_step_to_goal = evluation(w, tiling)
    average_to_goal.append(average_step_to_goal)

average_to_goal = np.mean(average_to_goal)
total_steps /= 10
plt.plot(y, total_steps)
plt.xlabel("Episodes")
plt.ylabel("Number of Actions")
plt.title(f'N Step Sarsa Cartpole \n alpha = {best_alpha} n = {best_n} gamma = {best_gamma} \n Average steps per run: {average_to_goal}')
plt.savefig("n_Cartpole_good.png")
plt.legend()
plt.close()

best_alpha = 0.0625
best_n = 4
best_gamma = 0.9
epsilon = 0.1
num_tile = 8

from tqdm import tqdm

env = gym.make("CartPole-v1")
y = range(0, 350)
total_steps = np.zeros(350)
average_to_goal = []
for _ in tqdm(range(10)):
    w, tiling, _, steps = episodic_Sarsa(350, env, best_alpha, best_n, best_gamma)
    total_steps += steps
    average_step_to_goal = evluation(w, tiling)
    average_to_goal.append(average_step_to_goal)

average_to_goal = np.mean(average_to_goal)
total_steps /= 10
plt.plot(y, total_steps)
plt.xlabel("Episodes")
plt.ylabel("Number of Actions")
plt.title(f'N Step Sarsa Cartpole \n alpha = {best_alpha} n = {best_n} gamma = {best_gamma} \n Average steps per run: {average_to_goal}')
plt.savefig("n_Cartpole_bad.png")
plt.legend()
plt.close()