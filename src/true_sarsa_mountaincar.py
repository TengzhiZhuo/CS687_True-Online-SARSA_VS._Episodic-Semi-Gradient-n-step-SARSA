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
        self.position_scale = self.num_tile / (0.5 + 1.2)
        self.velocity_scale = self.num_tile / (0.07 + 0.07)

    def get_tiles(self, position, velocity, action):
        return np.array(tile.tiles(self.iht, self.num_tile, [position*self.position_scale, velocity*self.velocity_scale], [action]))

def find_active_tiles(state, action, tile):
    x_sa = tile.get_tiles(state[0], state[1], action)
    active_tiles = np.zeros(4096,)
    for elem in x_sa:
        active_tiles[elem] = 1.0
    return active_tiles

def q(state, tile, w):
    q_values = []
    for action in range(3):
        active_tiles = find_active_tiles(state, action, tile)
        q_values.append(np.dot(w, active_tiles))
    action = np.argmax(q_values)
    return action, q_values[action]

def true_online_sarsa(episodes, env, alpha, lam, gamma):
    w = np.zeros(num_tile**4,)
    tile = GridTiling(4096, num_tile)
    steps = []
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()[0]
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() 
        else:
            action, value = q(state, tile, w)
        x = find_active_tiles(state, action, tile)
        z = np.zeros(num_tile**4,)
        Q_old = 0
        step = 0
        while True:
            next_state, reward, terminated, truncated, info = env.step(action)
            if truncated:
                total_steps = step
                break
            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample() 
            else:
                next_action, value = q(next_state, tile, w)
            if terminated:
                next_x = np.zeros(num_tile**4)
            else:
                next_x = find_active_tiles(next_state, next_action, tile)
            Q = np.dot(w, x)
            next_Q = np.dot(w, next_x)
            delta = reward + gamma*next_Q - Q
            z = gamma*lam*z + (1-alpha*gamma*lam*np.dot(z, x))*x
            w += alpha*(delta+Q-Q_old)*z - alpha*(Q-Q_old)*x
            Q_old = next_Q
            x = next_x
            action = next_action      
            step += 1
            if terminated:
                total_steps = step
                break
        steps.append(total_steps)
    return w, tile, steps

def evluation(w, tiling):
    tile = tiling
    steps = []
    for _ in range(200):
        step = 0
        env = gym.make("MountainCar-v0")
        state = env.reset()[0]
        for i in range(200):
            action, _ = q(state, tile, w)
            next_state, _, terminated, _, _ = env.step(action)
            state = next_state
            step += 1
            if terminated:
                break
        env.close()
        steps.append(step)
    return np.mean(steps)

env = gym.make("MountainCar-v0")
lams = [0.96, 0.92, 0.84]
alpha = [0.3/8, 0.2/8, 0.5/8]
epsilon = 0.1
gamma = [1.0, 0.9, 0.95]
num_tile = 8

best_alpha = None
best_lam = None
best_gamma = None
best_steps = math.inf
for a in alpha:
    for lam in lams:
        for g in gamma:
            w, tiling, _ = true_online_sarsa(500, env, a, lam, g)
            steps = evluation(w, tiling)
            if steps <= best_steps:
                best_alpha = a
                best_lam = lam
                best_gamma = g
                best_steps = steps

print(best_alpha, best_lam, best_gamma)

best_alpha = 0.025
best_lam = 0.84
best_gamma = 1.0
epsilon = 0.1
num_tile = 8

env = gym.make("MountainCar-v0")
y = range(0, 500)
total_steps = np.zeros(500)
average_to_goal = []
for _ in range(10):
    w, tiling, steps = true_online_sarsa(500, env, best_alpha, best_lam, best_gamma)
    total_steps += steps
    average_step_to_goal = evluation(w, tiling)
    average_to_goal.append(average_step_to_goal)

average_to_goal = np.mean(average_to_goal)
total_steps /= 10
plt.plot(y, total_steps)
plt.xlabel("Episodes")
plt.ylabel("Number of Actions")
plt.title(f'True Online Sarsa MountainCar \n alpha = {best_alpha} lambda = {best_lam} gamma = {best_gamma} \n Average steps to goal: {average_to_goal}')
plt.savefig("true_mountaincar_good.png")
plt.legend()
plt.close()

best_alpha = 0.025
best_lam = 0.84
best_gamma = 0.9
epsilon = 0.1
num_tile = 8

env = gym.make("MountainCar-v0")
y = range(0, 500)
total_steps = np.zeros(500)
average_to_goal = []
for _ in range(10):
    w, tiling, steps = true_online_sarsa(500, env, best_alpha, best_lam, best_gamma)
    total_steps += steps
    average_step_to_goal = evluation(w, tiling)
    average_to_goal.append(average_step_to_goal)

average_to_goal = np.mean(average_to_goal)
total_steps /= 10
plt.plot(y, total_steps)
plt.xlabel("Episodes")
plt.ylabel("Number of Actions")
plt.title(f'True Online Sarsa MountainCar \n alpha = {best_alpha} lambda = {best_lam} gamma = {best_gamma} \n Average steps to goal: {average_to_goal}')
plt.savefig("true_mountaincar_bad.png")
plt.legend()
plt.close()