import random
import torch
import matplotlib.pyplot as plt
from gym import Env
import gym
from gym import spaces, make
import numpy as np
import itertools
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import math

import torch

print(torch.__version__)
print(torch.cuda.is_available())


def compute_Distance(x1, x2, y1, y2, heightt):
    d_h = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    d = ((x1 - x2)**2 + (y1 - y2)**2 + (heightt)**2)**0.5
    return d_h, d


def Coverage(heightt):
    angle_Cover = 45
    radians = math.radians(angle_Cover)
    tan_value = math.tan(radians)
    coverage = heightt * tan_value
    return coverage


def compute_rate(observation, allocation, height, numberChannel, numberIOT, BandWidth, Q_X, Q_Y):
    Rate = [[0 for i in range(numberChannel)]
            for t in range(numberIOT)]

    dis = np.zeros(numberIOT)
    dis_h = np.zeros(numberIOT)
    for i in range(numberIOT):
        dis_h[i], dis[i] = compute_Distance(
            observation[0],  Q_X[i], observation[1],  Q_Y[i], observation[2])
    if allocation != []:
        for i in range(len(allocation)):

            coverage = Coverage(observation[2])
            if dis_h[i] < coverage:
                V = 1 + parameter/dis[i]**pathLoss
                log = np.log2(V)
                Rate[int(allocation[i])][i] = BandWidth * log / 10**6
            else:
                Rate[int(allocation[i])][i] = 0

    return Rate, dis, dis_h


def Movement(old_X_UAV, old_Y_UAV, old_Z_UAV, counter_LR, counter_TD, counter_FB, size_env, step_size, height_step_size, action):

    observation = np.zeros(3, dtype=float)

    x = old_X_UAV
    y = old_Y_UAV
    z = old_Z_UAV

    if action == 0:
        observation[0] = x
        observation[1] = y
        observation[2] = z
    elif action == 1:
        if (x + step_size) <= size_env:
            observation[0] = x + step_size
            counter_LR = counter_LR + 1
        else:
            observation[0] = x
            counter_LR = counter_LR
        observation[1] = y
        observation[2] = z

    elif action == 2:
        if (x - step_size) >= 0:
            observation[0] = x - step_size
            counter_LR = counter_LR - 1
        else:
            observation[0] = x
            counter_LR = counter_LR
        observation[1] = y
        observation[2] = z

    elif action == 3:
        if (y + step_size) <= size_env:
            observation[1] = y + step_size
            counter_TD = counter_TD + 1
        else:
            observation[1] = y
            counter_TD = counter_TD
        observation[0] = x
        observation[2] = z

    elif action == 4:
        if (y - step_size) >= 0:
            observation[1] = y - step_size
            counter_TD = counter_TD - 1
        else:
            observation[1] = y
            counter_TD = counter_TD
        observation[0] = x
        observation[2] = z

    elif action == 5:
        if (z + height_step_size) <= ub_height:
            observation[2] = z + height_step_size
            counter_FB = counter_FB + 1
        else:
            observation[2] = z
            counter_FB = counter_FB
        observation[0] = x
        observation[1] = y

    elif action == 6:
        if (z - height_step_size) >= lb_height:
            observation[2] = z - height_step_size
            counter_FB = counter_FB - 1
        else:
            observation[2] = z
            counter_FB = counter_FB
        observation[0] = x
        observation[1] = y

    x = observation[0]
    y = observation[1]
    z = observation[2]

    return observation,  x, y, z, counter_LR, counter_TD, counter_FB


number_moves = 35


def Uer_Mobilityy(Q_x, Q_y, n_Episode, numberIoT, number_moves, size_env):

    xx = np.zeros(len(Q_x))
    yy = np.zeros(len(Q_x))
    for i in range(len(Q_x)):
        xx[i] = Q_x[i]
        yy[i] = Q_y[i]

    slot_user_mobility = [
        [-1 for _ in range(number_moves)] for _ in range(numberIoT)]

    for i in range(numberIoT):
        r = random.sample(range(50), number_moves)
        for j in range(number_moves):

            slot_user_mobility[i][j] = r[j]

    move_direction = [
        [-1 for _ in range(n_Episode)] for _ in range(numberIoT)]

    for i in range(numberIoT):
        for j in range(n_Episode):
            if j in slot_user_mobility[i]:
                move_direction[i][j] = random.randint(0, 4)

    state = np.zeros((n_Episode, numberIoT, int(size_env /
                                                step_size_move_User + 1), int(size_env/step_size_move_User + 1)))

    for j in range(n_Episode):
        for i in range(numberIoT):

            x = xx[i]
            y = yy[i]
            if i == 9 and j == 22:
                stop = 0
            if move_direction[i][j] >= 0:
                if move_direction[i][j] == 0 and (xx[i] + step_size_move_User) <= size_env:
                    xx[i] = xx[i] + step_size_move_User
                    x = xx[i]
                if move_direction[i][j] == 1 and (xx[i] - step_size_move_User) >= 0:
                    xx[i] = xx[i] - step_size_move_User
                    x = xx[i]
                if move_direction[i][j] == 2 and (yy[i] + step_size_move_User) <= size_env:
                    yy[i] = yy[i] + step_size_move_User
                    y = yy[i]
                if move_direction[i][j] == 3 and (yy[i] - step_size_move_User) >= 0:
                    yy[i] = yy[i] - step_size_move_User
                    y = yy[i]

            state[j][i][int(x/step_size_move_User)
                        ][int(y/step_size_move_User)] = 1

    trajectory_X = np.zeros((numberIoT, n_Episode))
    trajectory_Y = np.zeros((numberIoT, n_Episode))

    for i in range(numberIoT):
        for j in range(n_Episode):
            for x in range(int(size_env / step_size_move_User + 1)):
                for y in range(int(size_env / step_size_move_User + 1)):
                    if state[j][i][x][y] == 1:

                        trajectory_X[i][j] = x * step_size_move_User
                        trajectory_Y[i][j] = y * step_size_move_User

    return state, trajectory_X, trajectory_Y


SUMrate = []


SE = 0
Fairr = 0

flag = 0

path1 = 'file.txt'
file = open(path1, 'w')


lb_height = 50
ub_height = 550
height_step_size = 50

step_size = 50

parameter = 127900000
pathLoss = 3

Q_X_UAV = [0]
Q_Y_UAV = [0]
Q_Z_UAV = [lb_height]


def USER_movement(size_envv, numberIOT, step_sizev):
    Q_X1 = []
    Q_Y1 = []
    s = np.random.randint(
        int((size_envv+step_sizev) / step_sizev), size=(numberIOT, 2))
    for j in range(len(s)):
        Q_X1.append(s[j][0] * step_size)
        Q_Y1.append(s[j][1] * step_size)

    return Q_X1, Q_Y1


step_size_move_User = 100
numberIOT = 10


Q_Us_l = np.zeros((numberIOT, 2))
Q_UAV_l = np.zeros((1, 3))
Q_Us_h = np.zeros((numberIOT, 2)) + 500
Q_UAV_h = np.zeros((1, 3)) + 500
Q_UAV_h[0][2] = Q_UAV_h[0][2] + 50


class UAVEnv(Env):

    def __init__(self):

        self.height = 50
        self.numberIOT = numberIOT
        self.numberChannel = 6

        self.nodes = np.arange(self.numberIOT)

        self.P_allocation = list(
            itertools.combinations(self.nodes, self.numberChannel))

        self.slot = 0

        self.step_size = step_size
        self.size_env = 500
        self.n_Episode = 50

        self.height = 250
        self.lb_height = lb_height
        self.ub_height = ub_height
        self.height_step_size = height_step_size

        self.counter_LR = 0
        self.counter_TD = 0
        self.counter_FB = 0

        self.BandWidth = 200000

        self.SUM = 0

        self.R = np.zeros((self.n_Episode, self.numberIOT, self.numberChannel))

        self.Q_X, self.Q_Y = USER_movement(
            self.size_env, self.numberIOT, self.step_size)

        global Q_Xx, Q_Yy

        Q_Xx, Q_Yy = self.Q_X, self.Q_Y

        self.state, trajectory_X_users, trajectory_Y_users = Uer_Mobilityy(
            self.Q_X, self.Q_Y, self.n_Episode, numberIOT, number_moves, self.size_env)

        self.old_X_UAV = 0
        self.old_Y_UAV = 0
        self.old_Z_UAV = self.lb_height

        self.state = np.array([0, 0])

        self.action_space = spaces.MultiDiscrete(
            [7, 3, len(self.P_allocation)])

        spac = {
            'USEr': spaces.Box(low=0, high=1, shape=(10, 2), dtype=np.float64),
            'UAV': spaces.Box(low=0, high=11000, shape=(1, 3), dtype=np.float64),
            'slot': spaces.Box(low=0, high=50, shape=(1,), dtype=np.float64),
            'USErate': spaces.Box(low=0, high=500, shape=(10, 1), dtype=np.float64),

        }
        self.observation_space = spaces.Dict(spac)

    def step(self, action):
        info = {}

        obs = {
            'USEr': np.zeros((10, 2)),
            'UAV': np.zeros((1, 3)),
            'slot': np.zeros((1,)),
            'USErate': np.zeros((10, 1))
        }

        if self.slot == 4:
            stop = 0

        self.slot = self.slot + 1

        self.obs = self.state[self.slot-1]

        trajectory_X = np.zeros((self.numberIOT))
        trajectory_Y = np.zeros((self.numberIOT))

        for i in range(self.numberIOT):
            for x in range(int(self.size_env / step_size_move_User + 1)):
                for y in range(int(self.size_env / step_size_move_User + 1)):
                    if self.obs[i][x][y] == 1:

                        trajectory_X[i] = x * step_size_move_User
                        trajectory_Y[i] = y * step_size_move_User

        observation, oldd_X_UAV, oldd_Y_UAV, oldd_Z_UAV, counterr_LR, counterr_TD, counterr_FB = Movement(
            self.old_X_UAV, self.old_Y_UAV, self.old_Z_UAV, self.counter_LR, self.counter_TD, self.counter_FB, self.size_env, self.step_size, self.height_step_size, action[0])

        a = (abs(counterr_LR) + abs(counterr_TD) + abs(counterr_FB))
        b = (self.n_Episode - self.slot)

        if a > b:

            if action[1] == 0:
                Act = 2
            if action[1] == 1:
                Act = 4
            if action[1] == 2:
                Act = 6

            observation, self.old_X_UAV, self.old_Y_UAV, self.old_Z_UAV, self.counter_LR, self.counter_TD, self.counter_BF = Movement(
                self.old_X_UAV, self.old_Y_UAV, self.old_Z_UAV, self.counter_LR, self.counter_TD, self.counter_FB, self.size_env, self.step_size, self.height_step_size, Act)

        else:

            self.counter_LR = counterr_LR
            self.counter_TD = counterr_TD
            self.counter_FB = counterr_FB

            self.old_X_UAV = oldd_X_UAV
            self.old_Y_UAV = oldd_Y_UAV
            self.old_Z_UAV = oldd_Z_UAV

        Rate = [[0 for i in range(self.numberChannel)]
                for t in range(self.numberIOT)]

        diss = np.zeros(self.numberIOT)

        self.allocation = self.P_allocation[action[2]]
        if flag == 1:
            print(self.allocation)

        Rate, dis, dis_h = compute_rate(observation, self.allocation, self.height,
                                        self.numberChannel, self.numberIOT, self.BandWidth, trajectory_X, trajectory_Y)
        max_dis = ((self.size_env)**2 + (self.size_env)
                   ** 2 + (self.ub_height)**2)**0.5
        norm_dis = dis/max_dis

        sum_Rate = np.sum(Rate)
        self.R[self.slot-1] = Rate

        sum_r = []
        sum_r = np.sum(self.R, axis=2)
        sum_ri = np.sum(sum_r, axis=0)
        SUMrate.append(np.sum(sum_r, axis=1))

        W = (np.sum(sum_ri))**2
        Z = np.sum(np.power(sum_ri, 2))

        if Z != 0:
            Fairness = W / (self.numberIOT * Z)
        else:
            Fairness = 0

        self.SUM += sum_Rate

        d = ((observation[0] - 0)**2 + (observation[1] - 0)
             ** 2 + (observation[2]-self.lb_height)**2)**0.5

        for i in (0, 1, 2):
            obs["UAV"][0][i] = observation[i]/500

        obs["UAV"][0][2] = observation[2]/550

        for i in range(self.numberIOT):
            obs["USEr"][i][0] = self.Q_X[i]/500
            obs["USEr"][i][1] = self.Q_Y[i]/500

        for i in range(self.numberIOT):
            obs["USErate"][i][0] = sum_ri[i]/(2*self.slot)

        obs["slot"][0] = self.slot/50

        reward = (sum_Rate * 1 + Fairness*2) - (d/(51-self.slot)**3)/2

        if self.slot == self.n_Episode:

            done = True

            if math.isnan(reward):
                stop = 0

            if reward > 0:
                reward = reward
            if flag == 1:

                print("rew: ", reward)
                print("SE :", sum_Rate)
                print("SUm :", self.SUM)
                print("Fairness :", Fairness)

        else:
            done = False

        Q_X_UAV.append(observation[0])
        Q_Y_UAV.append(observation[1])
        Q_Z_UAV.append(observation[2])

        file.write(" %d\r\n")
        file.write(" ---SLOT --- "+str(self.slot)+" --------------------- ")
        file.write(" ---SUM RATE --- "+str(sum_Rate)+" --- ")
        file.write(" ---FAIRNESS --- "+str(Fairness)+" --- ")
        file.write(" %d\r\n")
        file.write(
            " ------------------------------------------------------------------------- ")

        return obs, reward, done, info

    def render(self, _):

        pass

    def reset(self):

        obs = {
            'USEr': np.zeros((10, 2)),
            'UAV': np.zeros((1, 3)),
            'slot': np.zeros((1,)),
            'USErate': np.zeros((10, 1))
        }

        self.slot = 0
        self.R = np.zeros((self.n_Episode, self.numberIOT, self.numberChannel))

        self.old_X_UAV = 0
        self.old_Y_UAV = 0
        self.old_Z_UAV = self.lb_height

        self.counter_LR = 0
        self.counter_TD = 0
        self.counter_FB = 0
        self.SUM = 0
        self.Q_X, self.Q_Y = USER_movement(
            self.size_env, self.numberIOT, self.step_size)

        self.state, trajectory_X_users, trajectory_Y_users = Uer_Mobilityy(
            self.Q_X, self.Q_Y, self.n_Episode, numberIOT, number_moves, self.size_env)

        self.obs = self.state[self.slot]

        global Q_Xx, Q_Yy

        for i in range(self.numberIOT):
            obs["USEr"][i][0] = self.Q_X[i]/500
            obs["USEr"][i][1] = self.Q_Y[i]/500

        Q_Xx, Q_Yy = self.Q_X, self.Q_Y

        return obs


env = UAVEnv()


check_env(env)

policy_kwargs = dict(
    net_arch=[dict(pi=[32, 32], vf=[32, 32])])
timestep = 1600000
model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.0003,
            tensorboard_log='./result/', ent_coef=0.001, clip_range=0.2)

policy = model.policy


model.learn(total_timesteps=timestep, log_interval=1)

weights = policy.parameters_to_vector()
np.savetxt('MBE_data.txt',
           weights, delimiter=',', fmt='%.8f')

weightsssA = np.loadtxt('MBE_data.txt',
                        delimiter=',', dtype=float)
print(weightsssA)

model.save("MBE_data")
env = model.get_env()

del model

model = PPO.load("MBE_data")

print('befor_Weight')
print('after_Weight')
print(weightsssA)


Q_X_UAV = [0]
Q_Y_UAV = [0]
Q_Z_UAV = [lb_height]


obs = env.reset()

dones = False
sum_re = 0
actions = []
rewardssss = []
ii = 0
flag = 1

file.write(" %d\r\n")
file.write(
    " ------------------------------------------------------------------------- ")
file.write(
    " ------------------------------------------------------------------------- ")
file.write(
    " ------------------------------------------------------------------------- ")
file.write(" %d\r\n")
while not dones:
    SUMrate = []

    action, _states = model.predict(obs)
    print('episode: ' + str(ii))
    print(action)
    obs, rewards, dones, info = env.step(action)

    actions.append(action)
    rewardssss.append(rewards)

    print(rewards)
    ii = ii + 1

    env.render()

print(np.sum(rewardssss))

print(Q_X_UAV)
print(Q_Y_UAV)
print(Q_Z_UAV)
print(actions)


file.close()


dx = [Q_X_UAV[i+1] - Q_X_UAV[i] for i in range(len(Q_X_UAV)-1)]
dy = [Q_Y_UAV[i+1] - Q_Y_UAV[i] for i in range(len(Q_Y_UAV)-1)]
dz = [Q_Z_UAV[i+1] - Q_Z_UAV[i] for i in range(len(Q_Z_UAV)-1)]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Q_Xx, Q_Yy, 'bo')

ax.scatter([0], [0], [lb_height], s=70, linewidth=1, c='k',  marker='*')


for i in range(len(Q_X_UAV)-1):
    if dx[i] == 0 and dy[i] == step_size:
        ax.quiver(Q_X_UAV[i:i+1], Q_Y_UAV[i:i+1], Q_Z_UAV[i:i+1], dx[i:i+1], dy[i:i+1], dz[i:i+1],
                  color="#ff6a00",  alpha=0.6)

    if dx[i] == step_size and dy[i] == 0:
        ax.quiver(Q_X_UAV[i:i+1], Q_Y_UAV[i:i+1], Q_Z_UAV[i:i+1], dx[i:i+1], dy[i:i+1], dz[i:i+1],
                  color="#235e89",  alpha=0.6)

    if dx[i] == 0 and dy[i] == -step_size:
        ax.quiver(Q_X_UAV[i:i+1], Q_Y_UAV[i:i+1], Q_Z_UAV[i:i+1], dx[i:i+1], dy[i:i+1], dz[i:i+1],
                  color="#427f3b",  alpha=0.6)

    if dx[i] == -step_size and dy[i] == 0:
        ax.quiver(Q_X_UAV[i:i+1], Q_Y_UAV[i:i+1], Q_Z_UAV[i:i+1], dx[i:i+1], dy[i:i+1], dz[i:i+1],
                  color="#b81574",  alpha=0.6)

    if dz[i] == height_step_size:
        ax.quiver(Q_X_UAV[i:i+1], Q_Y_UAV[i:i+1], Q_Z_UAV[i:i+1], dx[i:i+1], dy[i:i+1], dz[i:i+1],
                  color="#b85574",  alpha=0.6)

    if dz[i] == -height_step_size:
        ax.quiver(Q_X_UAV[i:i+1], Q_Y_UAV[i:i+1], Q_Z_UAV[i:i+1], dx[i:i+1], dy[i:i+1], dz[i:i+1],
                  color="#b855f4",  alpha=0.6)


# Set axis labels
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')


plt.show()
