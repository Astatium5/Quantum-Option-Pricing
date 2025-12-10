from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
from qiskit.circuit.library import RYGate
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit.primitives import BackendSamplerV2
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import SamplerV2

try:
    from qiskit_ibm_runtime import Session, QiskitRuntimeService
except Exception:  # pragma: no cover - optional dependency guard
    Session = None
    QiskitRuntimeService = None  # type: ignore

try:  # optional typing only; keep import-time robust if module moves
    from qiskit_ibm_runtime.ibm_backend import IBMBackend
except Exception:  # pragma: no cover - optional dependency guard
    IBMBackend = None  # type: ignore


class _JobWrap:
    """Wraps a job to patch missing shots metadata in pub results."""

    def __init__(self, job, forced_shots: int):
        self._job = job
        self._shots = int(forced_shots)

    def result(self, *args, **kwargs):
        res = self._job.result(*args, **kwargs)
        try:
            for pubres in res:
                md = getattr(pubres, "metadata", None)
                if isinstance(md, dict) and "shots" not in md:
                    md["shots"] = self._shots
        except Exception:
            pass
        return res


class SamplerWithShots:
    """Force shots into SamplerV2 and normalize metadata."""

    def __init__(self, base_sampler, forced_shots: int):
        self._base = base_sampler
        self._shots = int(forced_shots)

    def run(self, pubs, **kwargs):
        kwargs.pop("shots", None)
        job = self._base.run(pubs, shots=self._shots, **kwargs)
        return _JobWrap(job, self._shots)


def sampler_with_forced_shots(backend, shots: int):
    """Convenience factory for Sampler (hardware-safe) with fixed shots."""
    shots = int(shots)

    def _is_ibm_backend(obj) -> bool:
        if IBMBackend is not None and isinstance(obj, IBMBackend):
            return True
        mod = getattr(obj, "__class__", type(obj)).__module__
        name = getattr(obj, "__class__", type(obj)).__name__.lower()
        return "qiskit_ibm_runtime" in str(mod) or "ibmbackend" in name

    # BackendSamplerV2 uses backend.run under the hood, which is removed on IBM hardware.
    # SamplerV2 routes through Runtime primitives. Open-plan accounts cannot use sessions,
    # so we fall back to sessionless SamplerV2(service=..., backend=...).
    if _is_ibm_backend(backend):
        base_sampler = None
        session = None
        if Session is not None:
            try:
                session = Session(backend=backend)
            except Exception:
                session = None

        if session is not None:
            base_sampler = SamplerV2(session=session)
        else:
            # Session-less path (for open/free plan). Provide backend via mode + options.
            name = getattr(backend, "name", None) or getattr(backend, "backend_name", None)
            # Newer qiskit-ibm-runtime SamplerOptions do NOT accept default_backend/backend.
            # Passing an empty dict keeps validation happy; the backend is already set via mode.
            opts = {}

            # Pass backend via mode to satisfy SamplerV2 requirements.
            base_sampler = SamplerV2(mode=backend, options=opts)
    else:
        base_sampler = BackendSamplerV2(backend=backend)

    return SamplerWithShots(base_sampler, forced_shots=shots)


def _extract_counts_from_pub(pub) -> dict:
    """Robustly extract counts dict {bitstring: count} from a Sampler v2 PubResult."""
    try:
        return pub.join_data().get_counts()
    except Exception:
        reg = next(iter(pub.data.values()))
        return reg.get_counts()


def binom_ci(p: float, shots: int, z: float = 1.96) -> Tuple[float, float]:
    shots = max(int(shots), 1)
    se = float(np.sqrt(max(p * (1 - p), 1e-12) / shots))
    lo = float(np.clip(p - z * se, 0.0, 1.0))
    hi = float(np.clip(p + z * se, 0.0, 1.0))
    return (min(lo, hi), max(lo, hi))


class CountingSampler:
    """Wrap a sampler to track total pubs submitted (for sweeps)."""

    def __init__(self, base, forced_shots: int):
        self.base = base
        self.forced_shots = int(forced_shots)
        self.reset()

    def reset(self):
        self.sampler_calls = 0
        self.pubs_total = 0

    def run(self, pubs, **kwargs):
        pubs_list = list(pubs)
        self.sampler_calls += 1
        self.pubs_total += len(pubs_list)
        kwargs.pop("shots", None)
        return self.base.run(pubs_list, shots=self.forced_shots, **kwargs)


@dataclass
class ThreeQComponents:
    measured: QuantumCircuit
    nominal: QuantumCircuit
    post_process: Callable[[float], float]
    p1_ideal: float
    objective_qubit: int


def build_3q_components(
    *,
    mu: float,
    sigma: float,
    low: float,
    high: float,
    strike_price: float,
    risk_free_rate: float,
    time_to_maturity: float,
    c_approx: float = 0.25,
) -> ThreeQComponents:
    """
    Construct the 3-qubit approximation circuit and post-processing used in the paper.
    Returns both measured and measurement-free versions plus helpers.
    """
    num_state = 2
    objective_qubit = 2
    low_f = float(np.asarray(low))
    high_f = float(np.asarray(high))

    grid = np.linspace(low_f, high_f, 2**num_state)
    payoffs = np.maximum(0.0, grid - float(strike_price))
    f_min = float(payoffs.min())
    f_max = float(payoffs.max())
    if (f_max - f_min) < 1e-12:
        raise RuntimeError("Degenerate payoff range for 3q. Check bounds/strike/spot.")

    tilde = 2.0 * (payoffs - f_min) / (f_max - f_min) - 1.0  # in [-1, 1]
    angles = 2.0 * (np.pi / 4.0 + c_approx * tilde)  # Ry angles

    discount = float(np.exp(-risk_free_rate * time_to_maturity))

    def post_process_from_p(p1: float) -> float:
        # First-order inversion used in the paper-style approximation
        e_tilde = (p1 - 0.5) / c_approx
        e_payoff = ((e_tilde + 1.0) / 2.0) * (f_max - f_min) + f_min
        # Clip to avoid absurd excursions, but do NOT zero-out negative values
        # (IAE noise can drive estimates slightly below the payoff range).
        e_payoff = float(np.clip(e_payoff, -0.25 * f_max, 1.25 * f_max))
        return discount * e_payoff

    def add_piecewise_ctrl_ry(qc: QuantumCircuit, ctrl_qubits, tgt, angle_list):
        n = len(ctrl_qubits)
        for j, theta in enumerate(angle_list):
            bits_lsb = format(j, f"0{n}b")[::-1]  # qubit0 is LSB in Qiskit indexing
            for k, b in enumerate(bits_lsb):
                if b == "0":
                    qc.x(ctrl_qubits[k])
            qc.append(RYGate(float(theta)).control(n), ctrl_qubits + [tgt])
            for k, b in enumerate(bits_lsb):
                if b == "0":
                    qc.x(ctrl_qubits[k])

    unc = LogNormalDistribution(num_state, mu=mu, sigma=sigma**2, bounds=(low_f, high_f))

    A3 = QuantumCircuit(3, 1)
    A3.compose(unc, qubits=[0, 1], inplace=True)
    add_piecewise_ctrl_ry(A3, ctrl_qubits=[0, 1], tgt=objective_qubit, angle_list=angles)
    A3.measure(objective_qubit, 0)

    A3_nom = A3.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(A3_nom)
    probs = sv.probabilities()  # ordering |q2 q1 q0|
    p1_ideal = float(np.sum(probs[4:8]))  # q2=1 -> indices 4..7

    return ThreeQComponents(
        measured=A3,
        nominal=A3_nom,
        post_process=post_process_from_p,
        p1_ideal=p1_ideal,
        objective_qubit=objective_qubit,
    )


def make_3q_problem(components: ThreeQComponents) -> EstimationProblem:
    return EstimationProblem(
        state_preparation=components.nominal,
        objective_qubits=[components.objective_qubit],
        post_processing=components.post_process,
    )

def run_iae_experiment(
    *,
    backend,
    pricing,
    problem: EstimationProblem,
    shots: int,
    epsilon: float,
    label: str,
    sampler_factory=None,
    pm=None,
    log_method: Optional[str] = None,
    log_fn: Optional[Callable[..., Any]] = None,
    use_counting_sampler: bool = False,
    alpha: float = 0.05,
) -> dict:
    pm = pm or None
    # Default sampler: hardware-safe with forced shots
    sampler_factory = sampler_factory or (lambda b: sampler_with_forced_shots(b, shots))
    base_sampler = sampler_factory(backend)

    tracker = None
    if use_counting_sampler:
        tracker = CountingSampler(base_sampler, forced_shots=shots)
        sampler = tracker
    else:
        sampler = base_sampler

    ae = IterativeAmplitudeEstimation(
        epsilon_target=float(epsilon),
        alpha=float(alpha),
        sampler=sampler,
        transpiler=pm,
    )
    res = ae.estimate(problem)
    interpret = getattr(pricing, "interpret", None)
    if interpret is None:
        raise ValueError("pricing must provide an interpret(res) method")
    est_price = float(interpret(res))
    try:
        ci_lo, ci_hi = map(float, res.confidence_interval_processed)
    except Exception:
        ci_lo, ci_hi = np.nan, np.nan

    if log_fn and log_method:
        log_fn(
            log_method,
            est_price,
            ci_lo,
            ci_hi,
            shots=int(shots),
            circuit_width=problem.state_preparation.num_qubits,
        )

    return {
        "label": label,
        "estimate": est_price,
        "ci": (ci_lo, ci_hi),
        "result": res,
        "pm": pm,
        "tracker": tracker,
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def direct_sample_problem_price(backend, pm, problem: EstimationProblem, shots: int):
    """
    Direct sampling matched to the SAME "good-event" as IAE.
    """
    shots = int(shots)
    obj = _get_objective_qubits(problem)
    A = problem.state_preparation

    qc = QuantumCircuit(A.num_qubits, len(obj))
    qc.compose(A, inplace=True)
    qc.measure(obj, list(range(len(obj))))

    qc_t = pm.run(qc)

    sampler = BackendSamplerV2(backend=backend)
    job = sampler.run([(qc_t,)], shots=shots)
    pubres = job.result()[0]

    counts_raw = pubres.join_data().get_counts()
    counts = {k.replace(" ", ""): int(v) for k, v in counts_raw.items()}

    key_ones = "1" * len(obj)
    k = int(counts.get(key_ones, 0))
    amp_hat = k / shots

    amp_lo, amp_hi = wilson_interval(k, shots, z=1.96)
    post = getattr(problem, "post_processing", None) or (lambda x: x)
    price_hat = float(post(amp_hat))
    price_lo = float(post(amp_lo))
    price_hi = float(post(amp_hi))

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    debug = {"k": k, "shots": shots, "amp_hat": float(amp_hat), "top_counts": top, "key_ones": key_ones}
    return price_hat, price_lo, price_hi, debug


def _get_objective_qubits(problem) -> List[int]:
    obj = getattr(problem, "objective_qubits", None)
    if obj is None:
        obj = getattr(problem, "objective_qubit", None)
    if obj is None:
        raise ValueError("Could not find objective_qubits/objective_qubit on the problem.")
    if isinstance(obj, int):
        obj = [obj]
    return [int(q) for q in obj]


def quantum_discretized_baseline(problem: EstimationProblem) -> float:
    """Exact expectation for the discretized quantum model."""
    A = getattr(problem, "state_preparation", None)
    if A is None:
        raise ValueError("problem.state_preparation not found")

    try:
        A_nom = A.remove_final_measurements(inplace=False)
    except Exception:
        A_nom = A

    obj = _get_objective_qubits(problem)
    sv = Statevector.from_instruction(A_nom)
    probs = np.asarray(sv.probabilities(), float)

    idx = np.arange(probs.shape[0], dtype=np.int64)
    cond = np.ones_like(idx, dtype=bool)
    for q in obj:
        cond &= ((idx >> q) & 1) == 1
    amp = float(probs[cond].sum())
    post = getattr(problem, "post_processing", None)
    return float(post(amp)) if callable(post) else float(amp)


def _extract_circuit_from_pub(pub):
    if hasattr(pub, "circuit"):
        return pub.circuit
    if isinstance(pub, (tuple, list)) and len(pub) > 0:
        return pub[0]
    return pub


class QueryCountingSamplerV2:
    """
    Wrapper around BackendSamplerV2 that:
      - injects shots if missing
      - counts A-calls budget: shots * (2*m + 1)
      - tries to infer m (Grover power) from metadata/name; falls back to depth heuristic
    """

    def __init__(self, base_sampler, default_shots: int):
        self.base = base_sampler
        self.default_shots = int(default_shots)
        self.reset()

    def reset(self):
        self.a_calls = 0.0
        self.circuits_total = 0
        self.shots_total = 0
        self.unknown_power_circuits = 0
        self.max_power = 0
        self._min_depth_seen = None
        self.unknown_examples = []

    def _infer_power(self, qc):
        md = getattr(qc, "metadata", None) or {}
        for k in ("grover_power", "power", "m", "k", "num_iterations", "iterations"):
            if k in md and md[k] is not None:
                try:
                    return max(int(md[k]), 0)
                except Exception:
                    pass

        name = str(getattr(qc, "name", "") or "")
        mm = re.search(r"(?:^|[^\\w])(m|power|k)\\s*=?\\s*(\\d+)", name, flags=re.IGNORECASE)
        if mm:
            try:
                return max(int(mm.group(2)), 0)
            except Exception:
                pass

        return None

    def run(self, pubs, **kwargs):
        shots = kwargs.get("shots", None)
        if shots is None:
            shots = self.default_shots
            kwargs["shots"] = shots
        shots = int(shots)

        pubs_list = pubs if isinstance(pubs, (list, tuple)) else [pubs]

        for pub in pubs_list:
            qc = _extract_circuit_from_pub(pub)

            depth = None
            try:
                depth = qc.depth()
            except Exception:
                pass
            if depth is not None:
                self._min_depth_seen = depth if self._min_depth_seen is None else min(self._min_depth_seen, depth)

            m = self._infer_power(qc)
            if m is None:
                if depth is not None and self._min_depth_seen is not None and depth <= 1.2 * self._min_depth_seen:
                    m = 0
                else:
                    self.unknown_power_circuits += 1
                    try:
                        self.unknown_examples.append({"name": qc.name, "depth": depth, "size": qc.size()})
                    except Exception:
                        self.unknown_examples.append({"name": str(getattr(qc, "name", "")), "depth": depth})
                    m = 0

            self.max_power = max(self.max_power, int(m))
            self.a_calls += shots * (2 * int(m) + 1)
            self.circuits_total += 1
            self.shots_total += shots

        return self.base.run(pubs, **kwargs)


def load_jsonl_robust(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                for p in line.replace("}{", "}\n{").splitlines():
                    p = p.strip()
                    if not p:
                        continue
                    try:
                        rows.append(json.loads(p))
                    except json.JSONDecodeError:
                        pass
    return pd.DataFrame(rows)


def fit_loglog_slope(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    b, _ = np.polyfit(np.log10(x), np.log10(y), 1)
    return float(b)


def summarize_rmse(dfi: pd.DataFrame, baseline: float) -> pd.DataFrame:
    dfi = dfi.copy()
    dfi["err"] = dfi["Estimate"] - float(baseline)
    dfi["abs_error"] = dfi["err"].abs()
    dfi["sq_error"] = dfi["err"] ** 2
    g = dfi.groupby(["family", "budget"], as_index=False)
    out = g.agg(
        rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
        p16=("abs_error", lambda s: float(np.quantile(s, 0.16))),
        p84=("abs_error", lambda s: float(np.quantile(s, 0.84))),
        n=("abs_error", "count"),
    ).sort_values(["family", "budget"]).reset_index(drop=True)
    return out


def slope_curve(M_ref: float, y_ref: float, exponent: float, M_min: float, M_max: float, num: int = 200):
    """Return (M, y) for a slope guide anchored at (M_ref, y_ref) with exponent."""
    M = np.logspace(np.log10(M_min), np.log10(M_max), num)
    y = y_ref * (M_ref / M) ** exponent
    return M, y
