"""CAS-PFSP 复现核心：实例解析、双随机键解码、适应度、模因算法。

对应论文：Mencaroni et al. (2025) "Towards net-zero manufacturing: carbon-aware
scheduling for GHG emissions reduction", J. Cleaner Production 529:146787.
实现依据 arXiv:2503.01325v2 第 2-3 节（MILP 模型 + MA-CAS-PFSP 算法）。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# 1. 实例解析
# --------------------------------------------------------------------------
@dataclass
class Instance:
    name: str
    M: int            # 机器数
    days: int         # 计划天数（每天 96 个 15min 时段）
    N: int            # 工件数
    T: int            # 时段总数
    D: list[list[int]]     # D[i][m] 加工时长（时段数）
    P: list[list[list[float]]]  # P[i][m][k] 第 k 时段功率需求（kWh/时段）
    A: np.ndarray     # 各时段可用厂内可再生电量 (kWh)
    C: np.ndarray     # 各时段电网碳强度 (gCO2eq/kWh)
    price: np.ndarray  # 各时段电价
    meta: dict = field(default_factory=dict)

    @property
    def total_slack(self) -> int:
        return sum(self.T - sum(self.D[i][m] for i in range(self.N)) for m in range(self.M))


def parse_instance(path: str) -> Instance:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    head = [float(x) for x in lines[0].replace(";", ",").split(",")]
    M, days, N = int(head[0]), int(head[1]), int(head[2])
    T = days * 96

    D: list[list[int]] = [[] for _ in range(N)]
    P: list[list[list[float]]] = [[] for _ in range(N)]
    idx = 1
    if M == 1:
        for i in range(N):
            vals = [float(x) for x in lines[idx].replace(";", ",").split(",") if x.strip() != ""]
            idx += 1
            D[i].append(len(vals))
            P[i].append(vals)
    else:
        # 多机算例每行前缀为 "工件号,机器号,"；空值表示该工序不在此机器加工
        for i in range(N):
            D[i] = [0] * M
            P[i] = [[] for _ in range(M)]
        for _ in range(N * M):
            parts = lines[idx].split(",")
            idx += 1
            i, m = int(parts[0]), int(parts[1])
            vals = [float(x) for x in parts[2:] if x.strip() != ""]
            D[i][m] = len(vals)
            P[i][m] = vals
    A = np.array([float(x) for x in lines[idx].replace(";", ",").split(",")])
    C = np.array([float(x) for x in lines[idx + 1].replace(";", ",").split(",")])
    price = np.array([float(x) for x in lines[idx + 2].replace(";", ",").split(",")])
    if len(A) != T:  # 兜底：以数据长度为准
        T = len(A)
    return Instance(name=path.split("\\")[-1], M=M, days=days, N=N, T=T,
                    D=D, P=P, A=A, C=C, price=price,
                    meta={"LB": head[-1] if len(head) >= 12 else None})


# --------------------------------------------------------------------------
# 2. 解码：双随机键 -> 各工序开始时段
# --------------------------------------------------------------------------
def _share(keys: np.ndarray, total: int) -> np.ndarray:
    """把 keys 按比例分配 total 个整数单位（最大余数法，保证和恰为 total）。"""
    k = np.asarray(keys, dtype=float)
    s = k.sum()
    if s <= 0 or total <= 0:
        out = np.zeros(len(k), dtype=int)
        if total > 0:
            out[-1] = total
        return out
    raw = k / s * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    if rem > 0:
        order = np.argsort(-(raw - base))
        base[order[:rem]] += 1
    elif rem < 0:
        order = np.argsort(raw - base)
        for j in order[:(-rem)]:
            if base[j] > 0:
                base[j] -= 1
    return base


def decode(inst: Instance, job_keys: np.ndarray, pause_keys: list[np.ndarray]):
    """返回 (start, feasible_len)；start[i][m] 为工序 (i,m) 开始时段。"""
    N, M, T, D = inst.N, inst.M, inst.T, inst.D
    seq = np.argsort(np.asarray(job_keys), kind="stable")
    start = np.zeros((N, M), dtype=int)
    makespan = 0
    for m in range(M):
        sum_d = sum(D[i][m] for i in range(N))
        slack = T - sum_d
        gaps = _share(pause_keys[m], slack) if slack > 0 else np.zeros(N + 1, dtype=int)
        free = 0
        for pos, i in enumerate(seq):
            s = free + int(gaps[pos])
            if m > 0:
                prev_end = start[i][m - 1] + D[i][m - 1]
                if s < prev_end:
                    s = prev_end
            start[i][m] = s
            free = s + D[i][m]
    makespan = int(max(start[i][m] + D[i][m] for i in range(N) for m in range(M)))
    return start, makespan


# --------------------------------------------------------------------------
# 3. 适应度：给定开始时段，计算负荷曲线与 Scope-2 排放
# --------------------------------------------------------------------------
def load_profile(inst: Instance, start: np.ndarray) -> np.ndarray:
    load = np.zeros(inst.T, dtype=float)
    for i in range(inst.N):
        for m in range(inst.M):
            s = int(start[i][m])
            d = inst.D[i][m]
            p = np.asarray(inst.P[i][m], dtype=float)
            end = min(s + d, inst.T)
            if end > s:
                load[s:end] += p[: end - s]
    return load


PERIOD_H = 0.25  # 每个时段 15 min = 0.25 h；论文结果（CSV）按此换算为 gCO2eq


def emissions_no_storage(inst: Instance, load: np.ndarray) -> float:
    """Scope-2 排放（gCO2eq）：可再生优先自用，余量从电网购电。"""
    grid = np.maximum(0.0, load - inst.A)
    return float(PERIOD_H * np.dot(grid, inst.C))


def fitness(inst: Instance, start: np.ndarray, makespan: int, cost_fn=None) -> float:
    load = load_profile(inst, start)
    obj = cost_fn(load) if cost_fn is not None else emissions_no_storage(inst, load)
    if makespan > inst.T:
        obj += (makespan - inst.T) * 1e10
    return obj


# --------------------------------------------------------------------------
# 4. 个体与种群操作
# --------------------------------------------------------------------------
class Individual:
    __slots__ = ("jk", "pk", "fit", "start", "makespan", "load")

    def __init__(self, jk, pk):
        self.jk = np.asarray(jk, dtype=float)
        self.pk = [np.asarray(x, dtype=float) for x in pk]
        self.fit = math.inf
        self.start = None
        self.makespan = 0
        self.load = None

    def copy(self):
        ind = Individual(self.jk.copy(), [x.copy() for x in self.pk])
        ind.fit, ind.makespan = self.fit, self.makespan
        ind.start = None if self.start is None else self.start.copy()
        return ind


def _normalize(a: np.ndarray) -> np.ndarray:
    s = a.sum()
    return a / s if s > 0 else np.full_like(a, 1.0 / len(a))


def _repair(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = np.where(a < 0, 0.0, a)
    if a.sum() <= 0:
        a = np.ones_like(a)
    return a / a.sum()


def evaluate(ind: Individual, inst: Instance, cost_fn=None, mk_cap=None):
    start, mk = decode(inst, ind.jk, ind.pk)
    ind.start, ind.makespan = start, mk
    ind.load = load_profile(inst, start)
    ind.fit = cost_fn(ind.load) if cost_fn is not None else emissions_no_storage(inst, ind.load)
    horizon = inst.T if mk_cap is None else min(inst.T, mk_cap)
    if mk > horizon:
        ind.fit += (mk - horizon) * 1e10
    return ind


def init_population(inst: Instance, rho: int, rng: random.Random):
    pop = []
    for _ in range(rho - 1):
        jk = np.array([rng.random() for _ in range(inst.N)])
        pk = [np.array([rng.expovariate(1.0) for _ in range(inst.N + 1)]) for _ in range(inst.M)]
        pop.append(Individual(_normalize(jk), [x / x.sum() for x in pk]))
    # FCFS 种子：作业键升序，暂停键只在最后一格非零
    jk = np.arange(1, inst.N + 1, dtype=float)
    pk = [np.array([0.0] * inst.N + [1.0]) for _ in range(inst.M)]
    pop.append(Individual(_normalize(jk), pk))
    return pop


def crossover(p1: Individual, p2: Individual, chi_j: float, chi_p: float, rng: random.Random):
    c1 = p1.copy()
    c2 = p2.copy()
    for k in range(len(c1.jk)):
        if rng.random() < chi_j:
            c1.jk[k], c2.jk[k] = c2.jk[k], c1.jk[k]
    for m in range(len(c1.pk)):
        for k in range(len(c1.pk[m])):
            if rng.random() < chi_p:
                c1.pk[m][k], c2.pk[m][k] = c2.pk[m][k], c1.pk[m][k]
    c1.jk, c2.jk = _normalize(c1.jk), _normalize(c2.jk)
    c1.pk = [_normalize(x) for x in c1.pk]
    c2.pk = [_normalize(x) for x in c2.pk]
    return c1, c2


def mutate(ind: Individual, pi_j: float, pi_p: float, sig_j: float, sig_p: float, rng: random.Random):
    for k in range(len(ind.jk)):
        if rng.random() < pi_j:
            ind.jk[k] = max(0.0, ind.jk[k] + rng.gauss(0.0, sig_j))
    ind.jk = _normalize(ind.jk)
    for m in range(len(ind.pk)):
        arr = ind.pk[m]
        for k in range(len(arr)):
            if rng.random() < pi_p:
                arr[k] = max(0.0, arr[k] + rng.gauss(0.0, sig_p))
        ind.pk[m] = _normalize(arr)
    return ind


def local_search(ind: Individual, inst: Instance, cost_fn=None, mk_cap=None):
    """相邻成对交换，首次改进即停止（论文 3.5 节）。"""
    best_fit = ind.fit
    order = np.argsort(ind.jk, kind="stable")
    for pos in range(len(order) - 1):
        a, b = order[pos], order[pos + 1]
        trial = ind.copy()
        trial.jk[a], trial.jk[b] = trial.jk[b], trial.jk[a]
        evaluate(trial, inst, cost_fn, mk_cap)
        if trial.fit < best_fit - 1e-9:
            return trial
    return ind


# --------------------------------------------------------------------------
# 5. 模因算法主循环
# --------------------------------------------------------------------------
def ma_cas_pfsp(inst: Instance, rho=250, gamma=100, xi=0.6, chi_j=0.3, chi_p=0.3,
                pi_j=0.2, pi_p=0.15, sig_j=0.1, sig_p=0.2,
                seed=0, cost_fn=None, mk_cap=None, log=False):
    rng = random.Random(seed)
    np.random.seed(seed % (2 ** 31 - 1))
    pop = init_population(inst, rho, rng)
    for ind in pop:
        evaluate(ind, inst, cost_fn, mk_cap)
    best = min(pop, key=lambda x: x.fit).copy()

    for gen in range(gamma):
        offspring = []
        n_cross = int(round(xi * rho))
        while len(offspring) < n_cross:
            p1 = rng.choice(pop)
            p2 = rng.choice(pop)
            if p1 is p2:
                continue
            c1, c2 = crossover(p1, p2, chi_j, chi_p, rng)
            offspring.append(c1)
            if len(offspring) < n_cross:
                offspring.append(c2)
        for c in offspring:
            mutate(c, pi_j, pi_p, sig_j, sig_p, rng)
            evaluate(c, inst, cost_fn, mk_cap)
            c2 = local_search(c, inst, cost_fn, mk_cap)
            if c2 is not c:
                offspring[offspring.index(c)] = c2
        # 精英直接进入子代
        for ind in pop[: rho - len(offspring)]:
            offspring.append(ind.copy())
        pool = pop + offspring
        pool.sort(key=lambda x: x.fit)
        pop = pool[:rho]
        if pop[0].fit < best.fit:
            best = pop[0].copy()
        if log and gen % 20 == 0:
            print(f"  gen {gen:3d} best={best.fit:,.1f}")
    return best