
import numpy as np
import random
import itertools

import matplotlib.pyplot as plt
from numpy import exp

from numpy.random import rand

from random import randint
import time
import math

parameter = 127900000
pathLoss = 3
Counter = 1
noisePower = -70
transmisionPower = 1

ch_gain_1m = -60
timeSlot = 2
vMax = 15
numberIOT = 2
numberChannel = 1

dis_max = vMax * timeSlot


size_env = 500
step_size = 50
Episode_n = 50
start_point = 0

height = 250
lb_height = 50
ub_height = 550
height_step_size = 50

number_of_channel = [6]
number_of_IOT = [10]


number_moves = 20

first_term = 0
step_size_move_User = 25
num_terms = 20


counter_LR = 0
counter_TD = 0


BandWidth = 200000


def compute_Velocity(distance, time):
    v = distance/time
    return v


def compute_Distance(x1, x2, y1, y2, heightt):
    d_h = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    d = ((x1 - x2)**2 + (y1 - y2)**2 + (heightt)**2)**0.5
    return d_h, d


def compute_ch_gain(distance, ch_gain_1m):
    h = ch_gain_1m / distance**2
    return h


def Coverage(heightt):
    angle_Cover = 45
    radians = math.radians(angle_Cover)
    tan_value = math.tan(radians)
    coverage = heightt * tan_value
    return coverage


def Uer_Mobilityy(Q_x, Q_y, n_Episode, numberIoT, number_moves, Seed):

    np.random.seed(Seed)
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
                move_direction[i][j] = random.randint(0, 3)

    state = np.zeros((n_Episode, numberIoT, int(size_env /
                                                step_size_move_User + 1), int(size_env/step_size_move_User + 1)))

    for j in range(n_Episode):
        for i in range(numberIoT):

            x = Q_x[i]
            y = Q_y[i]
            if i == 9 and j == 22:
                stop = 0
            if move_direction[i][j] >= 0:
                if move_direction[i][j] == 0 and (Q_x[i] + step_size_move_User) <= size_env:
                    Q_x[i] = Q_x[i] + step_size_move_User
                    x = Q_x[i]
                if move_direction[i][j] == 1 and (Q_x[i] - step_size_move_User) >= 0:
                    Q_x[i] = Q_x[i] - step_size_move_User
                    x = Q_x[i]
                if move_direction[i][j] == 3 and (Q_y[i] + step_size_move_User) <= size_env:
                    Q_y[i] = Q_y[i] + step_size_move_User
                    y = Q_y[i]
                if move_direction[i][j] == 4 and (Q_y[i] - step_size_move_User) >= 0:
                    Q_y[i] = Q_y[i] - step_size_move_User
                    y = Q_y[i]

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


def choose_point(numberIOT, numberChannel):

    MMoVVe = np.zeros((Episode_n))

    Sls_point = []
    for n in range(Episode_n):
        bool = True
        while(bool):
            matrix = np.zeros((numberIOT, numberChannel))
            x = []
            for i in range(numberIOT):
                All_CHs = np.arange(numberChannel).tolist()
                remain = list(set(All_CHs) - set(x))
                if remain:
                    j = random.choice(remain)
                else:
                    break
                matrix[i][j] = 1
                x.append(j)

            bool = False
        move = MMoVVe[n]

        Sls_point.append([matrix, move])

    return Sls_point


def Validity_check(numberIOT, numberChannel, point):

    return point


def Choose_neighbor(numberIOT, numberChannel, point, initial_pos_UAV):
    np.random.seed(1)
    move = np.zeros((Episode_n))
    matrix = np.zeros((Episode_n, numberIOT, numberChannel))

    flag_allocation = 1
    while (flag_allocation):
        flag_allocation = 0
        for n in range(Episode_n):
            move[n] = point[n][1]
            for i in range(numberIOT):
                for j in range(numberChannel):
                    matrix[n][i][j] = point[n][0][i][j]

        n = randint(0, Episode_n-1)

        flag_path = True
        while(flag_path):
            flag_path = False

            for n in range(Episode_n):
                move[n] = point[n][1]
            CLR = 0
            CTD = 0
            CFB = 0
            right_f = []
            left_f = []
            top_f = []
            down_f = []
            front_f = []
            back_f = []

            for n in range(Episode_n):
                if move[n] == 1:
                    CLR = CLR + 1

                if move[n] == 2:
                    CLR = CLR - 1

                if move[n] == 3:
                    CTD = CTD + 1

                if move[n] == 4:
                    CTD = CTD - 1

                if move[n] == 5:
                    CFB = CFB + 1

                if move[n] == 6:
                    CFB = CFB - 1

                if CTD == (size_env - start_point) / step_size:
                    top_f.append(n)
                if CTD == - start_point:
                    down_f.append(n)

                if CLR == (size_env - start_point) / step_size:
                    right_f.append(n)
                if CLR == - start_point:
                    left_f.append(n)

                if CFB == (ub_height - lb_height) / height_step_size:
                    front_f.append(n)
                if CFB == 0:
                    back_f.append(n)

            L = []
            R = []
            T = []
            D = []
            F = []
            B = []
            S = []

            for n in range(Episode_n):
                if move[n] == 0:
                    S.append(n)
                if move[n] == 1:
                    R.append(n)
                if move[n] == 2:
                    L.append(n)
                if move[n] == 3:
                    T.append(n)
                if move[n] == 4:
                    D.append(n)
                if move[n] == 5:
                    F.append(n)
                if move[n] == 6:
                    B.append(n)

            N = random.randint(0, Episode_n - 1)

            if N-1 in right_f:
                Rndm = random.choice([0, 2, 3, 4, 5, 6])
            elif N-1 in left_f:
                Rndm = random.choice([0, 1, 3, 4, 5, 6])
            elif N-1 in top_f:
                Rndm = random.choice([0, 1, 2, 4, 5, 6])
            elif N-1 in down_f:
                Rndm = random.choice([0, 1, 2, 3, 5, 6])
            elif N-1 in front_f:
                Rndm = random.choice([0, 1, 3, 2, 4, 6])
            elif N-1 in back_f:
                Rndm = random.choice([0, 1, 2, 3, 4, 5])
            else:
                Rndm = random.choice([0, 1, 2, 3, 4, 5, 6])

            flg = True
            if move[N] == 1:
                while(flg):
                    M = random.choice(L)
                    if M != N:
                        flg = False

            if move[N] == 2:
                while(flg):
                    M = random.choice(R)
                    if M != N:
                        flg = False

            if move[N] == 3:
                while(flg):
                    M = random.choice(D)
                    if M != N:
                        flg = False

            if move[N] == 4:
                while(flg):
                    M = random.choice(T)
                    if M != N:
                        flg = False

            if move[N] == 5:
                while(flg):
                    M = random.choice(B)
                    if M != N:
                        flg = False

            if move[N] == 6:
                while(flg):
                    M = random.choice(F)
                    if M != N:
                        flg = False

            if move[N] == 0:
                while(flg):
                    M = random.choice(S)
                    if M != N:
                        flg = False

                if Rndm == 0:
                    Rndm = random.choice([1, 2, 3, 4, 5, 6])

                if Rndm == 1:
                    move[N] = 1
                    v = randint(0, 1)
                    if v == 1 and R:
                        r = random.choice(R)
                        move[r] = 0
                    else:
                        move[M] = 2

                if Rndm == 2:
                    move[N] = 2
                    v = randint(0, 1)
                    if v == 1 and L:
                        l = random.choice(L)
                        move[l] = 0
                    else:
                        move[M] = 1

                if Rndm == 3:
                    move[N] = 3
                    v = randint(0, 1)
                    if v == 1 and T:
                        t = random.choice(T)
                        move[t] = 0
                    else:
                        move[M] = 4

                if Rndm == 4:
                    move[N] = 4
                    v = randint(0, 1)
                    if v == 1 and D:
                        d = random.choice(D)
                        move[d] = 0
                    else:
                        move[M] = 3

                if Rndm == 5:
                    move[N] = 5
                    v = randint(0, 1)
                    if v == 1 and F:
                        t = random.choice(F)
                        move[t] = 0
                    else:
                        move[M] = 6

                if Rndm == 6:
                    move[N] = 6
                    v = randint(0, 1)
                    if v == 1 and B:
                        d = random.choice(B)
                        move[d] = 0
                    else:
                        move[M] = 5

                Rndm = 10

            if Rndm == 0:
                move[N] = 0
                move[M] = 0
            if Rndm == 1:
                move[N] = 1
                move[M] = 2
            if Rndm == 2:
                move[N] = 2
                move[M] = 1
            if Rndm == 3:
                move[N] = 3
                move[M] = 4
            if Rndm == 4:
                move[N] = 4
                move[M] = 3
            if Rndm == 5:
                move[N] = 5
                move[M] = 6
            if Rndm == 6:
                move[N] = 6
                move[M] = 5

            if N < M:
                if move[N] > move[M]:
                    tmp = move[N]
                    move[N] = move[M]
                    move[M] = tmp

            position = np.zeros((Episode_n, 3))
            position[0] = initial_pos_UAV
            for n in range(Episode_n):
                if n == 0:
                    if move[n] == 0:
                        position[n][0] = initial_pos_UAV[0]
                        position[n][1] = initial_pos_UAV[1]
                        position[n][2] = initial_pos_UAV[2]
                    elif move[n] == 1:
                        position[n][0] = initial_pos_UAV[0] + step_size
                        position[n][1] = initial_pos_UAV[1]
                        position[n][2] = initial_pos_UAV[2]
                    elif move[n] == 2:
                        position[n][0] = initial_pos_UAV[0] - step_size
                        position[n][1] = initial_pos_UAV[1]
                        position[n][2] = initial_pos_UAV[2]
                    elif move[n] == 3:
                        position[n][0] = initial_pos_UAV[0]
                        position[n][1] = initial_pos_UAV[1] + step_size
                        position[n][2] = initial_pos_UAV[2]
                    elif move[n] == 4:
                        position[n][0] = initial_pos_UAV[0]
                        position[n][1] = initial_pos_UAV[1] - step_size
                        position[n][2] = initial_pos_UAV[2]
                    elif move[n] == 5:
                        position[n][0] = initial_pos_UAV[0]
                        position[n][1] = initial_pos_UAV[1]
                        position[n][2] = initial_pos_UAV[2] + height_step_size
                    elif move[n] == 6:
                        position[n][0] = initial_pos_UAV[0]
                        position[n][1] = initial_pos_UAV[1]
                        position[n][2] = initial_pos_UAV[2] - height_step_size

                else:

                    if move[n] == 0:
                        position[n][0] = position[n-1][0]
                        position[n][1] = position[n-1][1]
                        position[n][2] = position[n-1][2]
                    elif move[n] == 1:
                        position[n][0] = position[n-1][0] + step_size
                        position[n][1] = position[n-1][1]
                        position[n][2] = position[n-1][2]
                    elif move[n] == 2:
                        position[n][0] = position[n-1][0] - step_size
                        position[n][1] = position[n-1][1]
                        position[n][2] = position[n-1][2]
                    elif move[n] == 3:
                        position[n][0] = position[n-1][0]
                        position[n][1] = position[n-1][1] + step_size
                        position[n][2] = position[n-1][2]
                    elif move[n] == 4:
                        position[n][0] = position[n-1][0]
                        position[n][1] = position[n-1][1] - step_size
                        position[n][2] = position[n-1][2]
                    elif move[n] == 5:
                        position[n][0] = position[n-1][0]
                        position[n][1] = position[n-1][1]
                        position[n][2] = position[n-1][2] + height_step_size
                    elif move[n] == 6:
                        position[n][0] = position[n-1][0]
                        position[n][1] = position[n-1][1]
                        position[n][2] = position[n-1][2] - height_step_size

            if position[n][0] < 0 or position[n][0] > size_env or position[n][1] < 0 or position[n][1] > size_env or position[n][2] < lb_height or position[n][2] > ub_height:
                flag_path = True

        dis_h = np.zeros(numberIOT)

        in_coverage = [[]]*Episode_n
        for i in range(Episode_n):
            in_coverage[i] = []

        allocated = [[]]*Episode_n
        for i in range(Episode_n):
            allocated[i] = []

        for n in range(Episode_n):
            for i in range(numberIOT):
                for j in range(numberChannel):
                    dis_h[i], Dis = compute_Distance(
                        position[n][0],  Q_X[i], position[n][1],  Q_Y[i], position[n][2])
                    coverage = Coverage(position[n][2])

                    if matrix[n][i][j] == 1:
                        allocated[n].append(i)

                if dis_h[i] < coverage:
                    in_coverage[n].append(i)

        NN = random.choice([N, M])

        NoAllocat_InCover = [[]]*Episode_n
        for i in range(Episode_n):
            NoAllocat_InCover[i] = []
        for n in range(Episode_n):
            for i in in_coverage[n]:
                if i not in allocated[n]:
                    NoAllocat_InCover[n].append(i)

        Allocated_notINCover = [[]]*Episode_n
        for i in range(Episode_n):
            Allocated_notINCover[i] = []
        for n in range(Episode_n):
            for i in allocated[n]:
                if i not in in_coverage[n]:
                    Allocated_notINCover[n].append(i)

        for i in range(numberIOT):
            if i in Allocated_notINCover[NN]:
                for j in range(numberChannel):
                    if matrix[NN][i][j] == 1:
                        matrix[NN][i][j] = 0

        if len(in_coverage[NN]) <= numberChannel:
            for C in in_coverage[NN]:
                matrix[NN][C][1] = 1

        else:

            if allocated[NN]:
                ra = random.choice(allocated[NN])
                for j in range(numberChannel):
                    matrix[NN][ra][j] = 0

            rc = random.choice(NoAllocat_InCover[NN])
            matrix[NN][rc][1] = 1
            NoAllocat_InCover[NN].remove(rc)

    points = []
    for n in range(Episode_n):
        points.append([matrix[n], move[n]])

    return points


def simulated_annealing(utilitySA, n_iterations, temp, old_position, numberChannel, numberIOT, trajectory_X, trajectory_Y):
    # generate an initial point

    initial_pos_UAV = [0, 0, lb_height]

    best = Validity_check(numberIOT, numberChannel,
                          choose_point(numberIOT, numberChannel))

    # evaluate the initial point
    best_eval, _, _, _, _ = utilitySA(
        best, old_position, numberChannel, numberIOT, trajectory_X, trajectory_Y)
    # current working solution
    curr, curr_eval = best, best_eval
    scores = list()

    for i in range(n_iterations):

        candidate = Validity_check(
            numberIOT, numberChannel, Choose_neighbor(numberIOT, numberChannel, curr, initial_pos_UAV))

        # evaluate candidate point
        candidate_eval, _, _, _, _ = utilitySA(
            candidate, old_position, numberChannel, numberIOT, trajectory_X, trajectory_Y)

        # check for new best solution
        if candidate_eval <= best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            curr = best

            # report progress

        scores.append(best_eval)

        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)
        # check if we should keep the new point
        randd = rand()
        if diff < 0 or randd < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
    return [best, best_eval, scores]


def utilitySA(point, old_position, numberChannel, numberIOT, trajectory_X, trajectory_Y):
    U = []
    ff = 0
    flag2 = 0

    old_position = [start_point, start_point, lb_height]
    positions = np.zeros((Episode_n, 3))
    R = np.zeros((Episode_n, numberIOT, numberChannel))
    SUM = 0
    for n in range(Episode_n):

        positions[n] = old_position

        allocation = point[n][0]
        movement = point[n][1]

        position = [0, 0, 0]

        if movement == 0:
            position[0] = old_position[0]
            position[1] = old_position[1]
            position[2] = old_position[2]
        elif movement == 1:
            position[0] = old_position[0] + step_size
            position[1] = old_position[1]
            position[2] = old_position[2]
        elif movement == 2:
            position[0] = old_position[0] - step_size
            position[1] = old_position[1]
            position[2] = old_position[2]
        elif movement == 3:
            position[0] = old_position[0]
            position[1] = old_position[1] + step_size
            position[2] = old_position[2]
        elif movement == 4:
            position[0] = old_position[0]
            position[1] = old_position[1] - step_size
            position[2] = old_position[2]
        elif movement == 5:
            position[0] = old_position[0]
            position[1] = old_position[1]
            position[2] = old_position[2] + height_step_size
        elif movement == 6:
            position[0] = old_position[0]
            position[1] = old_position[1]
            position[2] = old_position[2] - height_step_size

        old_position[0] = position[0]
        old_position[1] = position[1]
        old_position[2] = position[2]

        if n == 49:
            stop = 0

        if position[0] < 0 or position[0] > size_env or position[1] < 0 or position[1] > size_env or position[2] < lb_height or position[2] > ub_height:

            return 100, positions, 0, 0, 0

        if n == 12:
            stop = 0

        Rate = [[0 for i in range(numberChannel)]
                for t in range(numberIOT)]

        d_Rate = [0 for i in range(numberIOT)]
        M_TT = [0 for i in range(numberChannel)]

        COVERR = []
        # comput_Rate
        dis_h = np.zeros(number_of_IOT)
        for i in range(numberIOT):
            for j in range(numberChannel):
                dis_h[i], Dis = compute_Distance(
                    position[0],  trajectory_X[i], position[1],  trajectory_Y[i], position[2])
                coverage = Coverage(position[2])
                if dis_h[i] < coverage:
                    COVERR.append(i)
                    if allocation[i][j] == 1:

                        if i == 1 and coverage >= 340:
                            stop = 0
                        V = 1 + parameter/Dis**pathLoss
                        log = np.log2(V)
                        Rate[i][j] = BandWidth * log / 10**6
                        test = 0
                        if Dis == 50:
                            stop = 0

                    else:
                        Rate[i][j] = 0

        sum_Rate = np.sum(Rate)
        R[n] = Rate

        sum_r = []
        sum_r = np.sum(R, axis=2)

        summmmmmm = np.sum(sum_r, axis=1)
        sum_ri = np.sum(sum_r, axis=0)
        W = (np.sum(sum_ri))**2
        Z = np.sum(np.power(sum_ri, 2))

        Fairness = W / (numberIOT * Z)

        if Z == 0:
            Fairness = 0
            stop = 0
        elif Fairness > 0:
            ff = Fairness
        if n == 37:
            stop = 0

        if movement == 0:
            energy = 0.00001
        else:
            energy = 0.1

        SUM = SUM + sum_Rate

    if flag2 == 1:
        return 0, positions, 0, 0, 0
    sum_U = (SUM) + Fairness * 100
    Rew = - sum_U

    return Rew, positions, summmmmmm, energy, ff


def USER_STArtPoint(size_envv, numberIOT, step_sizev, seed):
    np.random.seed(seed)
    Q_X1 = []
    Q_Y1 = []
    s = np.random.randint(
        int((size_envv+step_sizev) / step_sizev), size=(numberIOT, 2))
    for j in range(len(s)):
        Q_X1.append(s[j][0] * step_size)
        Q_Y1.append(s[j][1] * step_size)

    return Q_X1, Q_Y1


path1 = 'BEfix.txt'

current_time = 0
Counter = 1

list_err = np.zeros((Counter, 30))
# np.random.seed(1)
R_F_SA = np.zeros((len(number_of_channel), len(number_of_IOT)))
F_F_SA = np.zeros((len(number_of_channel), len(number_of_IOT)))
mul_SA = np.zeros((len(number_of_channel), len(number_of_IOT)))
flag = 0
for k in range(len(number_of_channel)):
    if k == 1:
        iii = 1
    for m in range(len(number_of_IOT)):

        SUMRATE_FSA = np.zeros(Counter)
        FAIRNESS_FSA = np.zeros(Counter)

        FAirnessList = []
        SumRateList = []
        UtltyList = []

        for counter in range(Counter):
            BEfix = open(path1, 'a')
            flag = 0

            print('!!!!!!!!!!! ' + str(counter))

            print('SA with number channel: ' +
                  str(number_of_channel[k]) + ' and number user: ' + str(number_of_IOT[m]))
            Q_X_UAV = []
            Q_Y_UAV = []
            Q_Z_UAV = []

            numberIOT = number_of_IOT[m]
            numberChannel = number_of_channel[k]

            R = np.zeros((Episode_n, numberIOT, numberChannel))

            np.random.seed(1)
            s = np.random.randint(size_env, size=(numberIOT, 2))

            nodes = np.arange(numberIOT)
            ch = np.arange(numberChannel)
            if numberChannel <= numberIOT:
                P_allocation = list(
                    itertools.permutations(nodes, numberChannel))
            else:
                P_allocation = list(
                    itertools.permutations(ch, numberIOT))
            all = []
            old_position = [start_point, start_point, lb_height]
            E = []
            start = time.time()
            random.seed(1)
            # define the total iterations
            n_iterations = 3000
            # initial temperature
            temp = 100
            SUM_SA = 0

            Q_X, Q_Y = USER_STArtPoint(
                size_env, numberIOT, step_size, counter)

            best, score, scores = simulated_annealing(
                utilitySA, n_iterations, temp, old_position, numberChannel, numberIOT, Q_X, Q_Y)
            print(
                '-------------------------------------------------TIME--------------------------------------------')
            current_time = current_time + time.time() - start
            print("Run Time: " + str(time.time() - start))
            print('--------------------------------------------')
            print(score)
            print('--------------------------------------------')

            for i2 in range(len(scores)):
                if (i2+1) % 100 == 0:
                    err = np.abs(scores[len(scores)-1] -
                                 scores[i2])/(scores[len(scores)-1])
                    list_err[counter][int(((i2+1) / 100)-1)] = -err

            flag = 1
            _, positions, sum_Rate, energy, fairnessSA = utilitySA(
                best, old_position, numberChannel, numberIOT, Q_X, Q_Y)
            print(str(_)+' : final best scor')
            print(str(np.sum(sum_Rate))+' : Sum Rate')
            print(str(fairnessSA)+' : fairness')

            print('111111111111111111111111111111111111111111111111111111111111111111')

            FAirnessList.append(fairnessSA)
            SumRateList.append(np.sum(sum_Rate))
            UtltyList.append(_)

            for n in range(Episode_n):
                Q_X_UAV.append(positions[n][0])
                Q_Y_UAV.append(positions[n][1])
                Q_Z_UAV.append(positions[n][2])
            print(Q_X)
            print(Q_Y)

            E.append(energy)

            SUMRATE_FSA[counter] = SUM_SA
            FAIRNESS_FSA[counter] = fairnessSA

            if (counter+1) % 5 == 0:

                print(sum(FAirnessList)/(counter+1))
                print(sum(SumRateList)/(counter+1))
                print(sum(UtltyList)/(counter+1))
                print(current_time)

                BEfix.write(" %d\r\n")
                BEfix.write(" ---number --- "+str(counter+1) +
                            " --------------------- ")
                BEfix.write(" ---Time --- "+str(current_time) +
                            " --------------------- ")
                BEfix.write(" ---SUM RATE --- " +
                            str(sum(SumRateList)/(counter+1))+" --- ")
                BEfix.write(" ---FAIRNESS --- " +
                            str(sum(FAirnessList)/(counter+1))+" --- ")
                BEfix.write(" %d\r\n")
                BEfix.write(
                    " ------------------------------------------------------------------------- ")
                BEfix.close()

        R_F_SA[k][m] = np.sum(SUMRATE_FSA) / Counter
        F_F_SA[k][m] = np.sum(FAIRNESS_FSA) / Counter
        mul_SA[k][m] = R_F_SA[k][m] * F_F_SA[k][m]

        print(sum(FAirnessList)/(Counter))
        print(sum(SumRateList)/(Counter))
        print(sum(UtltyList)/(Counter))


Q_X_UAV.append(0)
Q_Y_UAV.append(0)
Q_Z_UAV.append(lb_height)

sum_err = np.sum(list_err, axis=0)
avg_err = sum_err/Counter

plt.plot(avg_err, linewidth=2)
plt.xlabel('Iterations_BEFIX')
plt.ylabel('err')
plt.xlim((0, 30))
plt.ylim((0, 0.5))
plt.grid()
plt.show()

stop = 0
