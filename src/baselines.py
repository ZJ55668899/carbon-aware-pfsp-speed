"""对比算法：实数编码遗传算法（GA）与粒子群算法（PSO）。

两种算法与本文方法共用同一套双随机键编码与解码器，种群规模、迭代次数相同，
差别只在搜索算子，用于回答"收益是不是来自速度键而不是算子本身"。
"""
from __future__ import annotations

import random

import numpy as np

from cas_speed import Ind, SpeedOptions, evaluate, init_population


def _normalize(a):
    a = np.asarray(a, float)
    a = np.where(a < 0, 0.0, a)
    s = a.sum()
    return a / s if s > 0 else np.full_like(a, 1.0 / len(a))


def run_ga(inst, opts: SpeedOptions, rho=120, gamma=60, seed=0,
           pc=0.9, pm=0.1, sigma=0.1, cost_fn=None):
    rng = random.Random(seed)
    pop = init_population(inst, rho, rng, opts)
    for ind in pop:
        evaluate(ind, inst, opts, cost_fn)
    n_gene = inst.N * (1 + inst.M + 1)  # 作业键 + 暂停键 + 速度键（近似）

    for _ in range(gamma):
        pop.sort(key=lambda x: x.fit)
        new = [pop[0].copy(), pop[1].copy()]  # 精英保留
        while len(new) < rho:
            p1 = min(rng.sample(pop, 2), key=lambda x: x.fit)
            p2 = min(rng.sample(pop, 2), key=lambda x: x.fit)
            c = p1.copy()
            if rng.random() < pc:
                for k in range(len(c.jk)):
                    if rng.random() < 0.5:
                        c.jk[k] = p2.jk[k]
                for m in range(len(c.pk)):
                    for k in range(len(c.pk[m])):
                        if rng.random() < 0.5:
                            c.pk[m][k] = p2.pk[m][k]
                for k in range(len(c.sk)):
                    if rng.random() < 0.5:
                        c.sk[k] = p2.sk[k]
            for k in range(len(c.jk)):
                if rng.random() < pm:
                    c.jk[k] = max(0.0, c.jk[k] + rng.gauss(0, sigma))
            c.jk = _normalize(c.jk)
            for m in range(len(c.pk)):
                arr = c.pk[m]
                for k in range(len(arr)):
                    if rng.random() < pm:
                        arr[k] = max(0.0, arr[k] + rng.gauss(0, sigma))
                c.pk[m] = _normalize(arr)
            for k in range(len(c.sk)):
                if rng.random() < pm:
                    c.sk[k] = min(0.999, max(0.0, c.sk[k] + rng.gauss(0, sigma)))
            evaluate(c, inst, opts, cost_fn)
            new.append(c)
        pop = new
    pop.sort(key=lambda x: x.fit)
    return pop[0]


def run_pso(inst, opts: SpeedOptions, rho=120, gamma=60, seed=0,
            w=0.7, c1=1.5, c2=1.5, vmax=0.2, cost_fn=None):
    rng = random.Random(seed)
    n_j, n_p, n_s = inst.N, inst.M * (inst.N + 1), inst.N * inst.M

    def flat(ind):
        return np.concatenate([ind.jk, *ind.pk, ind.sk])

    def build(vec):
        jk = _normalize(vec[:n_j])
        pk, pos = [], n_j
        for _ in range(inst.M):
            pk.append(_normalize(vec[pos:pos + inst.N + 1]))
            pos += inst.N + 1
        sk = np.clip(vec[pos:pos + n_s], 0, 0.999)
        return Ind(jk, pk, sk)

    pop = init_population(inst, rho, rng, opts)
    for ind in pop:
        evaluate(ind, inst, opts, cost_fn)
    X = np.array([flat(ind) for ind in pop])
    dim = X.shape[1]
    V = np.array([[rng.uniform(-vmax, vmax) for _ in range(dim)] for _ in range(rho)])
    pbest, pbest_fit = X.copy(), np.array([ind.fit for ind in pop])
    g = int(np.argmin(pbest_fit))
    gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]

    for _ in range(gamma):
        for i in range(rho):
            V[i] = w * V[i] + c1 * rng.random() * (pbest[i] - X[i]) + c2 * rng.random() * (gbest - X[i])
            V[i] = np.clip(V[i], -vmax, vmax)
            X[i] = X[i] + V[i]
            X[i][:n_j] = np.clip(X[i][:n_j], 1e-6, None)
            ind = build(X[i])
            evaluate(ind, inst, opts, cost_fn)
            if ind.fit < pbest_fit[i]:
                pbest[i], pbest_fit[i] = X[i].copy(), ind.fit
                if ind.fit < gbest_fit:
                    gbest, gbest_fit = X[i].copy(), ind.fit
    best = build(gbest)
    evaluate(best, inst, opts, cost_fn)
    return best