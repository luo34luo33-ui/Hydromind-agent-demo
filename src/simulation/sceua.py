import numpy as np
import re


def extract_params_from_code(code_str):
    params = set()
    for m in re.finditer(r'params\.get\s*\(\s*["\'](\w+)["\']', code_str):
        params.add(m.group(1))
    return sorted(params)


DEFAULT_BOUNDS = {
    "k": (0.05, 0.80),
    "k_fast": (0.10, 0.80),
    "k_slow": (0.001, 0.10),
    "k_base": (0.001, 0.05),
    "S0": (0.0, 200.0),
    "Smax": (0.0, 500.0),
    "S_min": (0.0, 50.0),
    "CN": (30.0, 95.0),
    "n_reservoirs": (1.0, 5.0),
    "alpha": (0.1, 0.99),
    "fc": (10.0, 200.0),
    "Ia": (0.0, 20.0),
    "a": (0.1, 0.99),
    "b": (0.01, 2.0),
    "c": (0.01, 2.0),
    "k_surface": (0.10, 0.80),
    "k_interflow": (0.01, 0.30),
    "k_groundwater": (0.001, 0.05),
    "threshold": (0.0, 100.0),
    "percolation": (0.01, 0.50),
}


def get_bounds_for_params(param_names):
    bounds = []
    for name in param_names:
        if name in DEFAULT_BOUNDS:
            bounds.append(DEFAULT_BOUNDS[name])
        else:
            lo = 0.01
            hi = min(max(100.0, 10.0 * lo), 1000.0)
            bounds.append((lo, hi))
    return bounds


def build_calibration_objective(code_str, precip, pet, q_obs, param_names, nse_func):
    def objective(x):
        sandbox = {"np": np}
        try:
            exec(code_str, sandbox)
        except Exception:
            return -1e10
        func = sandbox.get("simulate_runoff")
        if func is None:
            return -1e10
        param_dict = dict(zip(param_names, np.atleast_1d(x)))
        try:
            q_sim = np.asarray(func(precip, pet, param_dict), dtype=float)
        except Exception:
            return -1e10
        n = len(precip)
        if q_sim is None or len(q_sim) != n or not np.isfinite(q_sim).all():
            return -1e10
        return nse_func(q_obs, q_sim)

    return objective


class SCEUA:

    def __init__(self, bounds, objective_func, maxn=3000, p=2, m=None,
                 kstop=3, pcento=0.01, seed=21):
        self.n = len(bounds)
        self.bounds = np.array(bounds, dtype=float)
        self.objective = objective_func
        self.maxn = maxn
        self.p = p
        self.m = m if m is not None else max(2 * self.n + 1, 4)
        self.kstop = kstop
        self.pcento = pcento
        self.rng = np.random.default_rng(seed)

        self.s = self.p * self.m
        self.neval = 0
        self.history = []

    def _random_point(self):
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        return lo + self.rng.random(self.n) * (hi - lo)

    def _evaluate(self, x):
        self.neval += 1
        score = self.objective(np.array(x))
        return float(score) if np.isfinite(score) else -1e10

    def _sort_population(self, pop):
        pop.sort(key=lambda pt: -pt["score"])

    def _partition(self, pop):
        complexes = [[] for _ in range(self.p)]
        for i, pt in enumerate(pop):
            complexes[i % self.p].append(pt)
        return complexes

    def _trapezoidal_prob(self, num_points):
        p_i = np.array(
            [2.0 * (num_points + 1 - i) / (num_points * (num_points + 1))
             for i in range(1, num_points + 1)]
        )
        return p_i / p_i.sum()

    def _cce(self, points):
        n_pts = len(points)
        if n_pts < 3:
            return points

        q = min(self.n + 1, n_pts)
        beta = max(2 * self.n + 1, n_pts)

        for _ in range(beta):
            probs = self._trapezoidal_prob(n_pts)
            sel = np.sort(
                self.rng.choice(n_pts, size=min(q, n_pts), replace=False, p=probs)
            )
            sel_sorted = sorted(sel, key=lambda i: -points[i]["score"])

            idx_worst = sel_sorted[-1]
            x_worst = points[idx_worst]["x"].copy()
            f_worst = points[idx_worst]["score"]

            centroid = np.mean([points[i]["x"] for i in sel_sorted[:-1]], axis=0)

            x_reflect = 2 * centroid - x_worst
            better = False

            if self._within_bounds(x_reflect):
                f_reflect = self._evaluate(x_reflect)
                if f_reflect > f_worst:
                    points[idx_worst] = {"x": x_reflect, "score": f_reflect}
                    better = True
                else:
                    x_contract = (centroid + x_worst) / 2
                    f_contract = self._evaluate(x_contract)
                    if f_contract > f_worst:
                        points[idx_worst] = {"x": x_contract, "score": f_contract}
                        better = True

            if not better:
                x_rand = self._random_point()
                f_rand = self._evaluate(x_rand)
                points[idx_worst] = {"x": x_rand, "score": f_rand}

            self._sort_population(points)

        return points

    def _within_bounds(self, x):
        return bool(np.all((x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1])))

    def _initialize_population(self):
        pop = []
        for _ in range(self.s):
            x = self._random_point()
            score = self._evaluate(x)
            pop.append({"x": x, "score": score})
        self._sort_population(pop)
        return pop

    def calibrate(self):
        pop = self._initialize_population()

        initial_best = pop[0]["score"]
        initial_range = pop[0]["score"] - pop[-1]["score"]
        last_best = initial_best
        best_sofar = initial_best
        best_x = pop[0]["x"].copy()
        no_improve = 0

        while self.neval < self.maxn:
            complexes = self._partition(pop)

            evolved = []
            for c in complexes:
                evolved.append(self._cce(c))

            all_points = []
            for c in evolved:
                all_points.extend(c)
            self._sort_population(all_points)
            pop = all_points

            current_best = pop[0]["score"]

            if current_best > best_sofar:
                best_sofar = current_best
                best_x = pop[0]["x"].copy()

            self.history.append((int(self.neval), float(best_sofar)))

            improvement = abs(current_best - last_best)
            threshold = self.pcento * (abs(initial_best) + 1e-10)
            if improvement < threshold:
                no_improve += 1
            else:
                no_improve = 0

            if no_improve >= self.kstop:
                break

            current_range = pop[0]["score"] - pop[-1]["score"]
            if initial_range > 1e-10 and current_range / initial_range < self.pcento:
                break

            last_best = current_best

        return best_x, best_sofar, self.history
