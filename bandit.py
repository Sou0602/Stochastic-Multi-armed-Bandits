import numpy as np
import os

class bandit:
    def __init__(self,instance,algorithm, randomseed , epsilon , horizon):
        self.instance = instance
        self.algorithm = algorithm
        self.randomSeed = randomseed
        self.epsilon = epsilon
        self.horizon = horizon

        np.random.seed(self.randomSeed)

    def true_means(self) :

        file_instance = open(self.instance, "r")
        lines = [float(line.rstrip('\n')) for line in file_instance]
        self.N = np.size(lines)
        self.truemeans = lines

        return lines

def epsilon_greedy(N,p_true,epsilon,horizon,randomSeed):
    np.random.seed(randomSeed)
    count = 0
    reg = 0
    p_list = []
    u_list = []
    arm_list = np.arange(1,N+1)
    for i in range(1, N+1):
        rprob = np.random.uniform(0, 1)
        if rprob <= p_true[i-1]:
            p_list = np.append(p_list,1)
        else :
            p_list= np.append(p_list, 0)
        u_list = np.append(u_list,1)
    while count < horizon:
        random_prob = np.random.random()
        if random_prob > epsilon:
            #Exploit
            prob_r = np.random.uniform(0,1)
            amax = np.argmax(p_list)
            if prob_r <= p_true[amax]:
                prob = 1
            else:
                prob = 0
            p_list[amax] = (p_list[amax]*u_list[amax] + prob) / (1+u_list[amax])
            u_list[amax] = 1 + u_list[amax]
            reg = reg + (np.max(p_true) - p_list[amax])
        elif random_prob <= epsilon:
            #Explore
            amax = np.argmax(p_list)
            arm_explore = np.delete(arm_list,amax)
            exp_i = np.random.choice(arm_explore)

            prob_r = np.random.uniform(0,1)
            if prob_r <= p_true[exp_i-1]:
                prob = 1
            else:
                prob = 0
            p_list[exp_i-1] = (p_list[exp_i-1]*u_list[exp_i-1] + prob) / (1+u_list[exp_i-1])
            u_list[exp_i-1] = 1 + u_list[exp_i-1]
            reg = reg + (np.max(p_true) - p_list[exp_i-1])

        count = count +1


    return p_list,reg

def ucb(N, p_true, horizon, randomSeed):
    np.random.seed(randomSeed)
    reg = 0
    count = 1
    p_list = []
    u_list = []
    arm_list = np.arange(1, N + 1)
    for i in range(1, N + 1):
        rprob = np.random.uniform(0, 1)
        if rprob < p_true[i - 1]:
            p_list = np.append(p_list, 1)
        else:
            p_list = np.append(p_list, 0)
        u_list = np.append(u_list, 1)
    while count <= horizon:
        ucb_list = []
        for i in range(0, np.size(p_list)):
            ucb_list = np.append(ucb_list,p_list[i] + np.sqrt(2 * np.log(count) / u_list[i]))
        umax = np.argmax(ucb_list)
        # Arm selected is umax
        prob_r = np.random.uniform(0, 1)
        if prob_r <= p_true[umax]:
            prob = 1
        else:
            prob = 0
        p_list[umax] = (p_list[umax] * u_list[umax] + prob) / (1 + u_list[umax])
        u_list[umax] = 1 + u_list[umax]
        reg = reg + (np.max(p_true) - p_list[umax])
        count = count + 1

    return p_list, reg


def kl_ucb(N,p_true,horizon,randomSeed):
    def find_q_ucb(t, u, pval):
        eps = 0.5
        min = pval
        max = 1
        q = (min + max) / 2
        eps_list = []
        kl = 0
        while eps > 0.0001:

            if q == pval:
                kl = 0
            elif q == 1:
                kl = np.inf
            elif pval == 0:
                kl = np.log(2)
            else:
                kl = pval * np.log(pval / q) + (1 - pval) * np.log((1 - pval) / (1 - q))

            if kl < (np.log(t) + 3 * np.log(np.log(t))) / u:
                eps = q
                min = q
                q = (min + max) / 2
                eps = abs(eps - q)
            elif kl > (np.log(t) + 3 * np.log(np.log(t))) / u:
                eps = q
                max = q
                q = (min + max) / 2
                eps = abs(eps - q)

            elif kl == (np.log(t) + 3 * np.log(np.log(t))) / u:
                eps = 0

            eps_list = np.append(eps_list, eps)

        return q
    np.random.seed(randomSeed)
    count = 2
    reg = 0
    p_list = []
    u_list = []
    arm_list = np.arange(1, N + 1)
    for i in range(1, N + 1):
        rprob = np.random.uniform(0, 1)
        if rprob <= p_true[i - 1]:
            p_list = np.append(p_list, 1)
        else:
            p_list = np.append(p_list, 0)
        u_list = np.append(u_list, 1)
    while count <= horizon+1:
        kl_ucb_list = []
        for i in range(0, np.size(p_list)):
            kl_ucb_list = np.append(kl_ucb_list,find_q_ucb(count,u_list[i],p_list[i]))

        umax = np.argmax(kl_ucb_list)
        # Arm selected is umax
        prob_r = np.random.uniform(0, 1)
        if prob_r <= p_true[umax]:
            prob = 1
        else:
            prob = 0
        p_list[umax] = (p_list[umax] * u_list[umax] + prob) / (1 + u_list[umax])
        u_list[umax] = 1 + u_list[umax]
        reg = reg + np.max(p_true) - p_list[umax]

        count = count + 1

    return p_list

def thompson_sampling(N,p_true,randomSeed, horizon):
    np.random.seed(randomSeed)
    count = 0
    reg = 0
    p_list = []
    s_list = []
    f_list = []
    for i in range(0, N):
        p_list = np.append(p_list,0)
        s_list = np.append(s_list,0)
        f_list = np.append(f_list,0)
    arm_list = np.arange(1, N + 1)

    while count < horizon:
        beta_list = []
        for i in range(0, N ):
            beta_list = np.append(beta_list,np.random.beta(s_list[i] +1 , f_list[i]+1))
        amax = np.argmax(beta_list)
        rprob = np.random.uniform(0,1)
        if rprob <= p_true[amax]:
            prob = 1
            s_list[amax] = s_list[amax] + 1
        else:
            prob = 0
            f_list[amax] = f_list[amax] + 1

        p_list[amax] = (p_list[amax] * (s_list[amax] + f_list[amax] - 1)+ prob) / (s_list[amax] + f_list[amax])

        reg = reg + np.max(p_true) - p_list[amax]
        count = count + 1

    return p_list , reg

def thompson_sampling_with_hint(N,p_true,horizon, randomSeed,ls_hint):
    np.random.seed(randomSeed)
    count = 0
    reg = 0
    p_list = []
    s_list = []
    f_list = []
    for i in range(0, N):
        p_list = np.append(p_list,0)
        s_list = np.append(s_list,0)
        f_list = np.append(f_list,0)
    arm_list = np.arange(1, N + 1)

    while count < 0.2*horizon:
        beta_list = []
        for i in range(0, N ):
            beta_list = np.append(beta_list,np.random.beta(s_list[i] +1 , f_list[i]+1))
        amax = np.argmax(beta_list)
        rprob = np.random.uniform(0,1)
        if rprob <= p_true[amax]:
            prob = 1
            s_list[amax] = s_list[amax] + 1
        else:
            prob = 0
            f_list[amax] = f_list[amax] + 1

        p_list[amax] = (p_list[amax] * (s_list[amax] + f_list[amax] - 1)+ prob) / (s_list[amax] + f_list[amax])
        reg = reg + np.max(p_true) - p_list[amax]
        count = count + 1

    u_list = s_list + f_list
    maxhint = np.max(ls_hint)
    K = 100000

    while count < horizon and count >= 0.2*horizon:
        dist = np.random.beta(K * maxhint, (1 - maxhint) * K)
        abs_reg = [np.abs(p - dist) for p in p_list] # reg min
        pmin = np.argmin(abs_reg)
        prob_r = np.random.uniform(0, 1)
        if prob_r <= p_true[pmin]:
            prob = 1
        else:
            prob = 0
        p_list[pmin] = (p_list[pmin] * u_list[pmin] + prob) / (1 + u_list[pmin])
        u_list[pmin] = 1 + u_list[pmin]
        reg = reg + np.max(p_true) - p_list[pmin]
        count = count + 1

    return p_list , reg


if __name__ == '__main__':
    path = 'F:\cs747-pa1'
    instance = input("--instance")
    Instance_path = os.path.join(path,instance)
    algorithm = input("--algorithm")
    randomSeed = input("--randomSeed")
    epsilon = input("--epsilon")
    horizon = input("--horizon")

    Bandit = bandit(Instance_path,algorithm,int(randomSeed),float(epsilon),float(horizon))
    Bandit.true_means()

    if algorithm == "epsilon-greedy":
        emp_p , reg = epsilon_greedy(Bandit.N,Bandit.true_means(),Bandit.epsilon,Bandit.horizon,Bandit.randomSeed)
    elif algorithm == "ucb":
        emp_p , reg = ucb(Bandit.N,Bandit.true_means(),Bandit.horizon,Bandit.randomSeed)
    elif algorithm == "kl-ucb":
        emp_p , reg = kl_ucb(Bandit.N,Bandit.true_means(),Bandit.horizon,Bandit.randomSeed)
    elif algorithm == "thompson-sampling":
        emp_p , reg = thompson_sampling(Bandit.N,Bandit.true_means(),Bandit.randomSeed,Bandit.horizon)
    elif algorithm == "thompson-sampling-with-hint":
        emp_p , reg = thompson_sampling_with_hint(Bandit.N,Bandit.true_means(),Bandit.horizon,Bandit.randomSeed,np.sort(Bandit.true_means()))

    print(Instance_path , ",",algorithm , ",",randomSeed ,",", epsilon , ",",horizon , ",",reg,'\n')

