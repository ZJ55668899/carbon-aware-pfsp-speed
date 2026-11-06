"""扩展模型：可变加工速度的碳感知置换流水车间调度（CAS-PFSP-S）。

在 Mencaroni et al. (2025) 模型基础上增加"速度档位"决策：
    工序 (i,m) 以速度 s 加工时：时长 d' = round(D_im / s)，各时段功率 p'(k) = p(k') * s^alpha
    （alpha 为速度-功率指数，取 2.5，等价于切削功率随主轴转速的幂律关系）
编码仍为双随机键：作业序列键 + 各机器暂停键 + 各工序速度键。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from cas_pfsp import Instance, PERIOD_H, _share, _normalize, load_profile


@dataclass
class SpeedOptions:
    """速度档位与功率模型参数。

    levels    : 可选速度档（相对基准速度的倍率）
    alpha     : 切削功率对速度的幂指数，p_cut ∝ s^alpha
    aux_share : 辅助功率（液压、冷却、控制等）占工序平均功率的比例，不随速度变化
    """
    levels: tuple = (0.8, 0.9, 1.0, 1.1, 1.2)
    alpha: float = 2.0
    aux_share: float = 0.30


def speed_index(key: float, opts: SpeedOptions) -> int:
    return min(len(opts.levels) - 1, max(0, int(key * len(opts.levels))))


def speed_of(key: float, opts: SpeedOptions) -> float:
    return opts.levels[speed_index(key, opts)]


def _cached_operation(inst, i, m, li, opts: SpeedOptions):
    """按 (算例, 工序, 速度档, 参数) 缓存重采样后的功率曲线，避免每次评价重复计算。"""
    cache = getattr(inst, "_sp_cache", None)
    if cache is None:
        cache = {}
        inst._sp_cache = cache
    key = (i, m, li, round(opts.alpha, 3), round(opts.aux_share, 3))
    hit = cache.get(key)
    if hit is None:
        hit = scaled_operation(inst.P[i][m], opts.levels[li], opts)
        cache[key] = hit
    return hit


def scaled_operation(P: list[float], s: float, opts: SpeedOptions):
    """返回 (时长, 功率序列)：按速度重采样原功率曲线并施加 s^alpha 缩放。"""
    D = len(P)
    if D == 0:
        return 0, np.zeros(0)
    new_len = max(1, int(round(D / s)))
    p = np.asarray(P, float)
    out = np.empty(new_len)
    for k in range(new_len):
        lo = int(math.floor(k * D / new_len))
        hi = max(lo + 1, int(math.ceil((k + 1) * D / new_len)))
        out[k] = p[lo:hi].mean()
    aux = opts.aux_share * p.mean()
    cut = np.maximum(0.0, out - aux)
    return new_len, aux + cut * (s ** opts.alpha)


def decode_speed(inst: Instance, job_keys, pause_keys, speed_keys, opts: SpeedOptions):
    N, M, T = inst.N, inst.M, inst.T
    seq = np.argsort(np.asarray(job_keys), kind="stable")
    dur = np.zeros((N, M), dtype=int)
    prof = [[None] * M for _ in range(N)]
    for i in range(N):
        for m in range(M):
            li = speed_index(float(speed_keys[i * M + m]), opts)
            d, p = _cached_operation(inst, i, m, li, opts)
            dur[i][m], prof[i][m] = d, p

    start = np.zeros((N, M), dtype=int)
    for m in range(M):
        sum_d = int(dur[:, m].sum())
        slack = T - sum_d
        gaps = _share(pause_keys[m], max(0, slack)) if slack > 0 else np.zeros(N + 1, dtype=int)
        free = 0
        for pos, i in enumerate(seq):
            st = free + int(gaps[pos])
            if m > 0:
                st = max(st, int(start[i][m - 1] + dur[i][m - 1]))
            start[i][m] = st
            free = st + dur[i][m]
    mk = int(max(start[i][m] + dur[i][m] for i in range(N) for m in range(M)))
    return start, dur, prof, mk


def load_profile_speed(inst: Instance, start, dur, prof) -> np.ndarray:
    load = np.zeros(inst.T, dtype=float)
    for i in range(inst.N):
        for m in range(inst.M):
            s = int(start[i][m])
            d = int(dur[i][m])
            end = min(s + d, inst.T)
            if end > s:
                load[s:end] += np.asarray(prof[i][m])[: end - s]
    return load


class Ind:
    __slots__ = ("jk", "pk", "sk", "fit", "start", "dur", "prof", "makespan", "load")

    def __init__(self, jk, pk, sk):
        self.jk = np.asarray(jk, float)
        self.pk = [np.asarray(x, float) for x in pk]
        self.sk = np.asarray(sk, float)
        self.fit = math.inf
        self.start = self.dur = self.prof = self.load = None
        self.makespan = 0

    def copy(self):
        c = Ind(self.jk.copy(), [x.copy() for x in self.pk], self.sk.copy())
        c.fit, c.makespan = self.fit, self.makespan
        c.start, c.dur, c.prof, c.load = self.start, self.dur, self.prof, self.load
        return c


def evaluate(ind: Ind, inst: Instance, opts: SpeedOptions, cost_fn=None):
    start, dur, prof, mk = decode_speed(inst, ind.jk, ind.pk, ind.sk, opts)
    ind.start, ind.dur, ind.prof, ind.makespan = start, dur, prof, mk
    ind.load = load_profile_speed(inst, start, dur, prof)
    if cost_fn is None:
        grid = np.maximum(0.0, ind.load - inst.A)
        ind.fit = PERIOD_H * float(np.dot(grid, inst.C))
    else:
        ind.fit = float(cost_fn(ind.load))
    if mk > inst.T:
        ind.fit += (mk - inst.T) * 1e10
    return ind


def init_population(inst, rho, rng, opts: SpeedOptions, init_solution=None):
    pop = []
    n_s = inst.N * inst.M
    for _ in range(rho - 1):
        jk = np.array([rng.random() for _ in range(inst.N)])
        pk = [np.array([rng.expovariate(1.0) for _ in range(inst.N + 1)]) for _ in range(inst.M)]
        sk = np.array([rng.random() for _ in range(n_s)])
        pop.append(Ind(_normalize(jk), [_normalize(x) for x in pk], sk))
    jk = np.arange(1, inst.N + 1, dtype=float)
    pk = [np.array([0.0] * inst.N + [1.0]) for _ in range(inst.M)]
    sk = np.full(n_s, 0.55)  # 速度键 0.55 -> 中间档（1.0）
    pop.append(Ind(_normalize(jk), pk, sk))
    if init_solution is not None:
        # 热启动：把固定速度最优解作为初始个体（速度键取中间档），保证不劣于基线
        base = Ind(init_solution.jk, [x for x in init_solution.pk], sk)
        pop[0] = base
    return pop


def crossover(p1, p2, chi_j, chi_p, chi_s, rng):
    c1, c2 = p1.copy(), p2.copy()
    for k in range(len(c1.jk)):
        if rng.random() < chi_j:
            c1.jk[k], c2.jk[k] = c2.jk[k], c1.jk[k]
    for m in range(len(c1.pk)):
        for k in range(len(c1.pk[m])):
            if rng.random() < chi_p:
                c1.pk[m][k], c2.pk[m][k] = c2.pk[m][k], c1.pk[m][k]
    for k in range(len(c1.sk)):
        if rng.random() < chi_s:
            c1.sk[k], c2.sk[k] = c2.sk[k], c1.sk[k]
    c1.jk, c2.jk = _normalize(c1.jk), _normalize(c2.jk)
    c1.pk = [_normalize(x) for x in c1.pk]
    c2.pk = [_normalize(x) for x in c2.pk]
    return c1, c2


def mutate(ind, pi_j, pi_p, pi_s, sig_j, sig_p, sig_s, rng):
    for k in range(len(ind.jk)):
        if rng.random() < pi_j:
            ind.jk[k] = max(0.0, ind.jk[k] + rng.gauss(0, sig_j))
    ind.jk = _normalize(ind.jk)
    for m in range(len(ind.pk)):
        arr = ind.pk[m]
        for k in range(len(arr)):
            if rng.random() < pi_p:
                arr[k] = max(0.0, arr[k] + rng.gauss(0, sig_p))
        ind.pk[m] = _normalize(arr)
    for k in range(len(ind.sk)):
        if rng.random() < pi_s:
            ind.sk[k] = min(0.999, max(0.0, ind.sk[k] + rng.gauss(0, sig_s)))
    return ind


def local_search(ind, inst, opts, cost_fn=None):
    order = np.argsort(ind.jk, kind="stable")
    for pos in range(len(order) - 1):
        a, b = order[pos], order[pos + 1]
        trial = ind.copy()
        trial.jk[a], trial.jk[b] = trial.jk[b], trial.jk[a]
        evaluate(trial, inst, opts, cost_fn)
        if trial.fit < ind.fit - 1e-9:
            return trial
    return ind


def ma_cas_speed(inst, opts: SpeedOptions, rho=120, gamma=60, xi=0.6,
                 chi_j=0.3, chi_p=0.3, chi_s=0.2, pi_j=0.2, pi_p=0.15, pi_s=0.15,
                 sig_j=0.1, sig_p=0.2, sig_s=0.15, seed=0, cost_fn=None,
                 freeze_pause=False, init_solution=None):
    rng = random.Random(seed)
    pop = init_population(inst, rho, rng, opts, init_solution)
    for ind in pop:
        evaluate(ind, inst, opts, cost_fn)
    best = min(pop, key=lambda x: x.fit).copy()
    best.start = min(pop, key=lambda x: x.fit).start
    for gen in range(gamma):
        off = []
        n_cross = int(round(xi * rho))
        while len(off) < n_cross:
            p1, p2 = rng.choice(pop), rng.choice(pop)
            if p1 is p2:
                continue
            c1, c2 = crossover(p1, p2, chi_j, chi_p, chi_s, rng)
            off.append(c1)
            if len(off) < n_cross:
                off.append(c2)
        for c in off:
            mutate(c, pi_j, pi_p, pi_s, sig_j, sig_p, sig_s, rng)
            if freeze_pause:  # 禁止插入空闲：暂停键固定为 FCFS 结构
                c.pk = [np.array([0.0] * inst.N + [1.0]) for _ in range(inst.M)]
            evaluate(c, inst, opts, cost_fn)
            c2 = local_search(c, inst, opts, cost_fn)
            if c2 is not c:
                off[off.index(c)] = c2
        for ind in pop[: rho - len(off)]:
            off.append(ind.copy())
        pool = pop + off
        pool.sort(key=lambda x: x.fit)
        pop = pool[:rho]
        if pop[0].fit < best.fit:
            best = pop[0].copy()
    return best