
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, Tuple
from itertools import combinations
from math import comb  # Python 3.8+
from scipy.spatial.distance import cdist
import numpy as np


def initialization(N: int, dim: int, ub, lb):
    """
    初始化种群
    """
    ub = np.atleast_1d(ub).astype(float)
    lb = np.atleast_1d(lb).astype(float)
    Boundary = ub.shape[0]

    if Boundary == 1:
        # 所有维度的边界相同
        x = np.random.rand(N, dim) * (ub[0] - lb[0]) + lb[0]
        new_lb = np.full(dim, lb[0])
        new_ub = np.full(dim, ub[0])
    else:
        # 每个维度单独的边界
        x = np.zeros((N, dim))
        for i in range(dim):
            x[:, i] = np.random.rand(N) * (ub[i] - lb[i]) + lb[i]
        new_lb = lb.copy()
        new_ub = ub.copy()

    return x, new_lb, new_ub


def UniformPoint(N: int, M: int):
    H1 = 1
    while comb(H1 + M - 1, M - 1) <= N:
        H1 += 1

    # 第一组点
    W_list = []
    for c in combinations(range(1, H1 + M), M - 1):
        c = np.array(c)
        row = c - np.arange(M - 1)
        W_list.append(row)
    W = np.vstack(W_list)
    W = np.hstack([W, np.full((W.shape[0], 1), H1)])
    W = (W - np.hstack([np.zeros((W.shape[0], 1)), W[:, :-1]])) / H1

    # 第二组点
    if H1 < M:
        H2 = 0
        while comb(H1 + M - 1, M - 1) + comb(H2 + M, M - 1) <= N:
            H2 += 1
        if H2 > 0:
            W2_list = []
            for c in combinations(range(1, H2 + M), M - 1):
                c = np.array(c)
                row = c - np.arange(M - 1)
                W2_list.append(row)
            W2 = np.vstack(W2_list)
            W2 = np.hstack([W2, np.full((W2.shape[0], 1), H2)])
            W2 = (W2 - np.hstack([np.zeros((W2.shape[0], 1)), W2[:, :-1]])) / H2
            W = np.vstack([W, W2 / 2 + 1 / (2 * M)])

    W = np.maximum(W, 1e-6)
    N_actual = W.shape[0]
    return W, N_actual


def boundaryCheck(X, Xmin, Xmax):
    """
    边界处理函数，将超出边界的个体调整到边界值
    """
    X = X.copy()
    Xmin = np.atleast_1d(Xmin).astype(float)
    Xmax = np.atleast_1d(Xmax).astype(float)

    for i in range(X.shape[0]):
        FU = X[i, :] > Xmax
        FL = X[i, :] < Xmin
        inside = ~(FU | FL)
        X[i, :] = X[i, :] * inside + Xmax * FU + Xmin * FL
    return X


def information_entropy(population, k=5):
    if len(population) < k + 1:
        return 0.0
    distances = cdist(population, population)
    np.fill_diagonal(distances, np.inf)
    nearest_distances = np.sort(distances, axis=1)[:, :k]
    H_total = 0.0
    for i in range(len(population)):
        dist_to_neighbors = nearest_distances[i]
        total_dist = np.sum(dist_to_neighbors)
        if total_dist > 0:
            p = dist_to_neighbors / total_dist
            H_total += -np.sum(p * np.log(p + 1e-10))
    return H_total / len(population)


def adaptive_mutation(population, mutation_strength, Xmin, Xmax):
    mutated = population + mutation_strength * np.random.normal(0, 1, population.shape)
    return boundaryCheck(mutated, Xmin, Xmax)


def dominates(x, y):
    """判断 x 是否支配 y"""
    return np.all(x <= y, axis=1) & np.any(x < y, axis=1)


def checkDomination(fitness):
    """检查解集中的支配关系"""
    Np = fitness.shape[0]
    dom_vector = np.zeros(Np, dtype=int)
    all_perm = list(combinations(range(Np), 2))
    all_perm = np.array(all_perm + [(j, i) for (i, j) in all_perm])
    d = dominates(fitness[all_perm[:, 0]], fitness[all_perm[:, 1]])
    dominated_particles = np.unique(all_perm[d, 1])
    dom_vector[dominated_particles] = 1
    return dom_vector


def NDSort(PopObj, *args):
    """非支配排序 (NDSort)"""
    N, M = PopObj.shape
    if len(args) == 1:
        nSort = args[0]
    else:
        PopCon, nSort = args
        Infeasible = np.any(PopCon > 0, axis=1)
        PopObj[Infeasible, :] = (np.max(PopObj, axis=0) +
                                 np.sum(np.maximum(0, PopCon[Infeasible, :]), axis=1).reshape(-1, 1))
    return ENS_SS(PopObj, nSort)


def ENS_SS(PopObj, nSort):
    PopObj_unique, indices, inverse = np.unique(PopObj, axis=0, return_index=True, return_inverse=True)
    Table = np.bincount(inverse)
    N, M = PopObj_unique.shape
    FrontNo = np.full(N, np.inf)
    MaxFNo = 0

    while np.sum(Table[FrontNo < np.inf]) < min(nSort, len(inverse)):
        MaxFNo += 1
        for i in range(N):
            if FrontNo[i] == np.inf:
                Dominated = False
                for j in range(i - 1, -1, -1):
                    if FrontNo[j] == MaxFNo:
                        m = 1
                        while m < M and PopObj_unique[i, m] >= PopObj_unique[j, m]:
                            m += 1
                        Dominated = m == M
                        if Dominated or M == 2:
                            break
                if not Dominated:
                    FrontNo[i] = MaxFNo
    FrontNo = FrontNo[inverse]
    return FrontNo, int(MaxFNo)


def EnvironmentalSelection(mompa_getMOFcn, Population, N, M, Z, Zmin):
    if Zmin is None or len(Zmin) == 0:
        Zmin = np.ones(M)

    Population_objs = np.vstack([np.asarray(mompa_getMOFcn(ind)).reshape(1, -1)
                                  for ind in Population])
    FrontNo, MaxFNo = NDSort(Population_objs, N)
    Next = FrontNo < MaxFNo
    Last = np.where(FrontNo == MaxFNo)[0]
    K = N - np.sum(Next)
    if K > 0:
        PopObj1 = Population_objs[Next, :]
        PopObj2 = Population_objs[Last, :]
        Choose = LastSelection(PopObj1, PopObj2, K, Z, Zmin)
        Next[Last[Choose]] = True
    return Population[Next, :]


def LastSelection(PopObj1, PopObj2, K, Z, Zmin):
    PopObj = np.vstack([PopObj1, PopObj2]) - Zmin.reshape(1, -1)
    N, M = PopObj.shape
    N1, N2 = PopObj1.shape[0], PopObj2.shape[0]
    NZ = Z.shape[0]

    # 标准化
    w = np.eye(M) + 1e-6
    Extreme = []
    for i in range(M):
        tmp = PopObj / w[i, :]
        rowmax = np.max(tmp, axis=1)
        Extreme.append(np.argmin(rowmax))
    Extreme = np.array(Extreme)

    try:
        Hyperplane = np.linalg.lstsq(PopObj[Extreme, :], np.ones((M, 1)), rcond=None)[0].flatten()
        a = 1.0 / Hyperplane
        if np.any(np.isnan(a)) or np.any(np.isinf(a)):
            raise np.linalg.LinAlgError
    except Exception:
        a = np.max(PopObj, axis=0)
    PopObj = PopObj / a.reshape(1, -1)


    Cosine = 1 - cdist(PopObj, Z, metric='cosine')
    normP = np.linalg.norm(PopObj, axis=1, keepdims=True)
    Distance = normP * np.sqrt(1 - Cosine**2)
    d = np.min(Distance, axis=1)
    pi = np.argmin(Distance, axis=1)

    rho = np.bincount(pi[:N1], minlength=NZ)
    Choose = np.zeros(N2, dtype=bool)
    Zchoose = np.ones(NZ, dtype=bool)
    d2 = d[N1:]
    pi2 = pi[N1:]

    while np.sum(Choose) < K:
        Temp = np.where(Zchoose)[0]
        min_rho = np.min(rho[Temp])
        Jmin = Temp[np.where(rho[Temp] == min_rho)[0]]
        j = np.random.choice(Jmin)
        I = np.where((~Choose) & (pi2 == j))[0]
        if I.size > 0:
            if rho[j] == 0:
                s = I[np.argmin(d2[I])]
            else:
                s = np.random.choice(I)
            Choose[s] = True
            rho[j] += 1
        else:
            Zchoose[j] = False
    return Choose


def mompa_Gaussian_search(Prey, SearchAgents_no, dim, ub, lb):
    ub = np.atleast_1d(ub).astype(float)
    lb = np.atleast_1d(lb).astype(float)
    Prey = Prey.copy()

    for i in range(SearchAgents_no):
        d = np.random.randint(dim)
        Prey[i, d] += (ub[d] - lb[d]) * np.random.randn()

    for i in range(Prey.shape[0]):
        Flag4ub = Prey[i, :] > ub
        Flag4lb = Prey[i, :] < lb
        inside = ~(Flag4ub | Flag4lb)
        Prey[i, :] = Prey[i, :] * inside + ub * Flag4ub + lb * Flag4lb
    return Prey



def IMORBMO(params: Dict[str, Any], MultiObj: Dict[str, Any], plot_each_gen: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    Xmin = np.array(MultiObj["var_min"]).reshape(-1)
    Xmax = np.array(MultiObj["var_max"]).reshape(-1)
    fobj: Callable = MultiObj["fun"]
    numObj = MultiObj["numOfObj"]
    T = int(params["maxgen"])
    N = int(params["Np"])
    D = int(MultiObj["nVar"])
    Xfood = np.zeros(D)
    Epsilon = 0.5

    # 初始化
    X, new_lb, new_ub = initialization(N, D, Xmax, Xmin)
    Z, NZ = UniformPoint(N, numObj)
    SearchAgents_no = N

    subfit_list = [np.asarray(fobj(X[idx, :])).reshape(1, -1) for idx in range(SearchAgents_no)]
    subfit = np.vstack(subfit_list)
    Zmin = np.min(subfit, axis=0)

    X = EnvironmentalSelection(fobj, X, SearchAgents_no, numObj, Z, Zmin)
    X_old = X.copy()
    H0 = information_entropy(X)

    # 初始化绘图
    if plot_each_gen:
        plt.ion()
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d' if numObj == 3 else None)

    # 主循环
    t = 1
    while t <= T:
        # 搜寻食物
        NEWx1 = np.zeros_like(X)
        for i in range(N):
            p = np.random.randint(2, 6)
            Xpmean = np.mean(X[np.random.permutation(N)[:p], :], axis=0)
            q = np.random.randint(10, N+1) if N >= 10 else N
            Xqmean = np.mean(X[np.random.permutation(N)[:q], :], axis=0)
            R1 = np.random.permutation(N)[0]
            if np.random.rand() < Epsilon:
                NEWx1[i, :] = X[i, :] + (Xpmean - X[R1, :]) * np.random.rand()
            else:
                NEWx1[i, :] = X[i, :] + (Xqmean - X[R1, :]) * np.random.rand()
        NEWx1 = boundaryCheck(NEWx1, Xmin, Xmax)

        CF = (1 - t / T) ** (2 * t / T)
        NEWx2 = np.zeros_like(X)
        for i in range(N):
            p = np.random.randint(2, 6)
            Xpmean = np.mean(NEWx1[np.random.permutation(N)[:p], :], axis=0)
            q = np.random.randint(10, N+1) if N >= 10 else N
            Xqmean = np.mean(NEWx1[np.random.permutation(N)[:q], :], axis=0)
            if np.random.rand() < Epsilon:
                NEWx2[i, :] = Xfood + CF * (Xpmean - NEWx1[i, :]) * np.random.randn(D)
            else:
                NEWx2[i, :] = Xfood + CF * (Xqmean - NEWx1[i, :]) * np.random.randn(D)
        NEWx2 = boundaryCheck(NEWx2, Xmin, Xmax)

        H_global = information_entropy(X)
        denom = max(H0, 1e-6)
        if H_global < H0:
            mutation_strength = 0.5 * np.exp(-H_global / denom)
        else:
            mutation_strength = 0.01 + 0.1 * (1 - (H_global - H0) / (1.0 - H0 + 1e-6))
        mutated_population = adaptive_mutation(X, mutation_strength, Xmin, Xmax)

        # 猎物搜索
        Prey_gau = mompa_Gaussian_search(X, SearchAgents_no, D, Xmax, Xmin)
        two_Prey = np.vstack([X_old, NEWx1, NEWx2, Prey_gau, mutated_population])
        two_loc = np.unique(two_Prey, axis=0)

        subfit1 = np.vstack([np.asarray(fobj(two_loc[idx, :])).reshape(1, -1) for idx in range(two_loc.shape[0])])
        subfit = subfit1
        Zmin = np.min(np.vstack([Zmin, np.min(subfit, axis=0)]), axis=0)

        X = EnvironmentalSelection(fobj, two_loc, SearchAgents_no, numObj, Z, Zmin)
        X_old = X.copy()

        subfit2 = np.vstack([np.asarray(fobj(X[i, :])).reshape(1, -1) for i in range(X.shape[0])])
        fit = subfit2.copy()
        DOMINATED = checkDomination(fit)
        fit_non_dominated = fit[DOMINATED == 0, :]

        print(
            f"迭代数：{t} - MORBMO 帕累托前沿解个数: {fit_non_dominated.shape[0]}, "
            f"mutation_strength: {mutation_strength:.4f}"
        )

        # 动态绘图
        if plot_each_gen:
            ax.cla()
            if numObj == 3:
                if "truePF" in MultiObj and MultiObj["truePF"] is not None:
                    pf = np.array(MultiObj["truePF"])
                    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], s=10, c='k')
                ax.scatter(fit_non_dominated[:, 0], fit_non_dominated[:, 1], fit_non_dominated[:, 2], s=30, c='g')
                ax.set_xlabel('f1'); ax.set_ylabel('f2'); ax.set_zlabel('f3')
            elif numObj == 2:
                if "truePF" in MultiObj and MultiObj["truePF"] is not None:
                    pf = np.array(MultiObj["truePF"])
                    ax.scatter(pf[:, 0], pf[:, 1], s=10, c='k')
                ax.scatter(fit_non_dominated[:, 0], fit_non_dominated[:, 1], s=30, c='g')
                ax.set_xlabel('f1'); ax.set_ylabel('f2')
            plt.title(MultiObj.get("name", "MORBMO"))
            plt.pause(0.001)

        t += 1

    # 最后一代结果绘制
    plt.ioff()
    fig = plt.figure(figsize=(6, 5))
    if numObj == 2:
        ax = fig.add_subplot(111)
        ax.scatter(fit_non_dominated[:, 0], fit_non_dominated[:, 1], s=30, c='g')
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
    elif numObj == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(fit_non_dominated[:, 0], fit_non_dominated[:, 1], fit_non_dominated[:, 2], s=30, c="g")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
    plt.title(MultiObj.get("name", "MORBMO"))
    plt.tight_layout()
    plt.show()

    return X.copy(), fit.copy()
