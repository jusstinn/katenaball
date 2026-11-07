"""D4 per-frame possession model: LightGBM, deterministic on CPU, with temperature calibration.

    model = Model(kind="twohead", features="all").fit(X, y, names)
    model.calibrate(raw_oof, y_oof)            # temperatures from out-of-fold raw scores
    logp = model.predict_logp(feats)            # float32[T, 3], log P(H, A, D), rows normalised
    model.save(path); Model.load(path)

kind="3class": one multiclass booster; log P = log_softmax(z / T).
kind="twohead": an alive/dead booster on every row (logit a = log P(alive) / P(dead)) and a team
booster trained on alive rows only (logit h = log P(H | alive) / P(A | alive)). With temperatures
T_a and T_h: log P(H) = log sig(a/T_a) + log sig(h/T_h), log P(A) = log sig(a/T_a) + log sig(-h/T_h),
log P(D) = log sig(-a/T_a).

Calibration is fitted on out-of-fold predictions of models trained without those rows (see
`katena.model.lomo`); `fit` alone leaves every temperature at 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize_scalar

from ..feats import ball_free
from ..match import A, D, H

MODEL_VERSION = "d4-lgbm-v1"

BASE_PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.6,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 63,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 6,
    "seed": 20261107,
    "verbose": -1,
}
DEFAULT_ROUNDS = 200


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def _fit_temperature(nll_of_T, lo: float = 0.2, hi: float = 5.0) -> float:
    res = minimize_scalar(lambda u: nll_of_T(float(np.exp(u))), bounds=(np.log(lo), np.log(hi)),
                          method="bounded", options={"xatol": 1e-4})
    return float(np.round(np.exp(res.x), 4))


def select_features(names: list, features) -> list:
    """'all', 'ball_free', or an explicit list of column names."""
    if isinstance(features, str):
        if features == "all":
            return list(names)
        if features == "ball_free":
            return ball_free(list(names))
        raise ValueError(f"unknown feature set {features!r}")
    missing = [c for c in features if c not in names]
    if missing:
        raise KeyError(f"features not available: {missing[:5]}")
    return list(features)


class Model:
    """LightGBM possession model with columns H, A, D in every output."""

    def __init__(self, kind: str = "twohead", features="all", params: dict | None = None,
                 rounds: int = DEFAULT_ROUNDS, rounds_team: int | None = None):
        if kind not in ("3class", "twohead"):
            raise ValueError("kind must be '3class' or 'twohead'")
        self.kind = kind
        self.features = features
        self.params = {**BASE_PARAMS, **(params or {})}
        self.rounds = int(rounds)
        self.rounds_team = int(rounds_team if rounds_team is not None else rounds)
        self.boosters: dict = {}
        self.columns: list = []
        self.temperature: dict = self._unit_temperature()
        self.info: dict = {}

    def _unit_temperature(self) -> dict:
        return {"T": 1.0} if self.kind == "3class" else {"T_alive": 1.0, "T_team": 1.0}

    # -- training ------------------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, names: list) -> "Model":
        """X float32[N, F] with column `names`, y int codes in {H, A, D}."""
        y = np.asarray(y)
        if not np.isin(y, (H, A, D)).all():
            raise ValueError("training labels must be H, A or D")
        self.columns = select_features(list(names), self.features)
        idx = [list(names).index(c) for c in self.columns]
        Xs = X if idx == list(range(X.shape[1])) else np.ascontiguousarray(X[:, idx])
        self.boosters = {}
        if self.kind == "3class":
            p = {**self.params, "objective": "multiclass", "num_class": 3}
            ds = lgb.Dataset(Xs, y.astype(np.int32), feature_name=self.columns, free_raw_data=True)
            self.boosters["multi"] = lgb.train(p, ds, num_boost_round=self.rounds)
        else:
            p = {**self.params, "objective": "binary"}
            alive = (y != D)
            ds = lgb.Dataset(Xs, alive.astype(np.int32), feature_name=self.columns, free_raw_data=True)
            self.boosters["alive"] = lgb.train(p, ds, num_boost_round=self.rounds)
            del ds
            Xa = Xs[alive]
            ds = lgb.Dataset(Xa, (y[alive] == H).astype(np.int32), feature_name=self.columns,
                             free_raw_data=True)
            self.boosters["team"] = lgb.train(p, ds, num_boost_round=self.rounds_team)
            del ds, Xa
        self.temperature = self._unit_temperature()
        self.info = {"n_train_rows": int(len(y)), "class_share": {
            k: float(np.mean(y == c)) for k, c in (("H", H), ("A", A), ("D", D))}}
        return self

    # -- inference -----------------------------------------------------------------------------
    def _matrix(self, X: np.ndarray, names: list) -> np.ndarray:
        if list(names) == self.columns:
            return X
        pos = {n: i for i, n in enumerate(names)}
        missing = [c for c in self.columns if c not in pos]
        if missing:
            raise KeyError(f"input lacks model columns: {missing[:5]}")
        return np.ascontiguousarray(X[:, [pos[c] for c in self.columns]])

    def raw_scores(self, X: np.ndarray, names: list) -> np.ndarray:
        """Uncalibrated margins: float64[T, 3] multiclass scores, or [T, 2] (alive, team) logits."""
        Xm = self._matrix(X, names)
        nt = int(self.params["num_threads"])
        if self.kind == "3class":
            return np.asarray(self.boosters["multi"].predict(Xm, raw_score=True, num_threads=nt),
                              dtype=np.float64)
        a = self.boosters["alive"].predict(Xm, raw_score=True, num_threads=nt)
        h = self.boosters["team"].predict(Xm, raw_score=True, num_threads=nt)
        return np.stack([a, h], axis=1).astype(np.float64)

    def logp_from_raw(self, raw: np.ndarray, temperature: dict | None = None) -> np.ndarray:
        t = temperature or self.temperature
        raw = np.asarray(raw, dtype=np.float64)
        if self.kind == "3class":
            out = _log_softmax(raw / t["T"])
        else:
            a, h = raw[:, 0] / t["T_alive"], raw[:, 1] / t["T_team"]
            la, ld = _log_sigmoid(a), _log_sigmoid(-a)
            out = np.stack([la + _log_sigmoid(h), la + _log_sigmoid(-h), ld], axis=1)
        return out.astype(np.float32)

    def predict_logp(self, feats) -> np.ndarray:
        """float32[T, 3]: log P(H), log P(A), log P(D) per row of `feats` (a `Feats`)."""
        return self.logp_from_raw(self.raw_scores(feats.X, feats.names))

    # -- calibration ---------------------------------------------------------------------------
    def fit_temperature(self, raw: np.ndarray, y: np.ndarray) -> dict:
        """Temperatures minimising NLL of out-of-fold raw scores `raw` (from `raw_scores`)."""
        y = np.asarray(y)
        raw = np.asarray(raw, dtype=np.float64)
        if self.kind == "3class":
            yi = y.astype(np.int64)

            def nll(T):
                lp = _log_softmax(raw / T)
                return -float(np.mean(lp[np.arange(len(yi)), yi]))
            t = {"T": _fit_temperature(nll)}
        else:
            alive = (y != D).astype(np.float64)
            a = raw[:, 0]

            def nll_a(T):
                z = a / T
                return -float(np.mean(alive * _log_sigmoid(z) + (1 - alive) * _log_sigmoid(-z)))
            m = y != D
            h, is_h = raw[m, 1], (y[m] == H).astype(np.float64)

            def nll_h(T):
                z = h / T
                return -float(np.mean(is_h * _log_sigmoid(z) + (1 - is_h) * _log_sigmoid(-z)))
            t = {"T_alive": _fit_temperature(nll_a), "T_team": _fit_temperature(nll_h)}
        return t

    def set_temperature(self, t: dict) -> "Model":
        expect = set(self._unit_temperature())
        if set(t) != expect:
            raise KeyError(f"temperature keys must be {sorted(expect)}")
        self.temperature = {k: float(v) for k, v in t.items()}
        return self

    # -- persistence ---------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": MODEL_VERSION, "kind": self.kind, "features": self.features,
                "params": self.params, "rounds": self.rounds, "rounds_team": self.rounds_team,
                "columns": self.columns, "temperature": self.temperature, "info": self.info,
                "boosters": {k: b.model_to_string() for k, b in sorted(self.boosters.items())}}

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        if d.get("version") != MODEL_VERSION:
            raise ValueError(f"model version {d.get('version')!r} != {MODEL_VERSION!r}")
        m = cls(kind=d["kind"], features=d["features"], params=d["params"], rounds=d["rounds"],
                rounds_team=d["rounds_team"])
        m.columns = list(d["columns"])
        m.boosters = {k: lgb.Booster(model_str=s) for k, s in d["boosters"].items()}
        m.temperature = {k: float(v) for k, v in d["temperature"].items()}
        m.info = d.get("info", {})
        return m

    @classmethod
    def load(cls, path) -> "Model":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
