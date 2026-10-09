"""Jev runtime: evaluates PyIF's System-1 primitives (jev.choice / jev.noul / jev.filter).

Each primitive is elaborated with the object graph (candidate labels, extra fields, entity rules), sent as
one batched System-1 request, and gated by confidence. Uncertain decisions go to an optional fallback
(normally one LLM call); without a fallback the best guess is used and marked ungated.
"""
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field

from .config import CACHE

NONE = "NONE"
LABEL_KEY = re.compile(r"(name|名称|简称|标题|title)$", re.I)


@dataclass
class Gate:
    """Choice: act iff top option != NONE and p_top >= choice_min (or the conformal set is a singleton).
    Noul: yes iff p >= noul_hi, no iff p <= noul_lo, otherwise uncertain."""
    choice_min: float = 0.6
    noul_hi: float = 0.7
    noul_lo: float = 0.3
    conformal_tau: float | None = None   # if set: act iff {c : 1 - p_c <= tau} == {top}
    enabled: bool = True


def conformal_tau(true_probs, alpha):
    """Split-conformal threshold from calibration scores 1 - p(true option)."""
    scores = sorted(1 - p for p in true_probs)
    n = len(scores)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return scores[k - 1]


@dataclass
class Decision:
    kind: str
    question: str
    options: dict
    probs: dict
    result: object
    gated: str            # "act" | "fallback" | "ungated"
    latency: float = 0.0
    tokens: int = 0
    extra: dict = field(default_factory=dict)


class JevRuntime:
    def __init__(self, sog=None, model="jev-1.13.0", elaboration="full", gate=None, ledger=None, fallback=None,
                 debias=1, use_cache=True, max_fields=6):
        from typesafe_sdk import TypeSafeClient
        self.client = TypeSafeClient(model=model)
        self.model, self.sog, self.elaboration = model, sog, elaboration
        self.gate, self.ledger, self.fallback = gate or Gate(), ledger, fallback
        self.debias, self.use_cache, self.max_fields = max(1, debias), use_cache, max_fields
        self.cache_dir = CACHE / "jev"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request, self.decisions, self.records = "", [], {}
        self.id_keys = sog.identifier_keys() if sog else {}
        self.label_names = {}
        if sog:
            for e in sog.objects.entities:
                p = sog.property(e.label_property) if e.label_property else None
                if p:
                    self.label_names[e.id] = p.name

    # ---------- run context ----------
    def start(self, request: str):
        self.request, self.decisions, self.records = request, [], {}

    def observe(self, value):
        """Remember every record that carries an identifier, so bare codes can be labelled later."""
        if isinstance(value, dict):
            k = self._id_key(value)
            if k is not None:
                self.records.setdefault(str(value[k]), value)
            for v in value.values():
                self.observe(v)
        elif isinstance(value, list):
            for v in value:
                self.observe(v)

    # ---------- elaboration ----------
    def _id_key(self, rec):
        return next((k for k in rec if k in self.id_keys and not isinstance(rec[k], (dict, list))), None)

    def _entity(self, rec):
        k = self._id_key(rec)
        return self.id_keys.get(k) if k else None

    def _label(self, rec):
        ent = self._entity(rec)
        name = self.label_names.get(ent)
        if name and rec.get(name):
            return str(rec[name])
        return next((str(v) for k, v in rec.items() if LABEL_KEY.search(k) and isinstance(v, str) and v), None)

    def _option(self, item, key_field=None):
        """(key, display text, compact record) for one candidate."""
        rec = item if isinstance(item, dict) else self.records.get(str(item))
        if isinstance(item, dict):
            k = key_field if key_field in item else self._id_key(item)
            key = str(item[k]) if k else json.dumps(item, ensure_ascii=False)[:60]
        else:
            key = str(item)
        if self.elaboration == "raw" or rec is None:
            return key, None, {"value": key}
        label = self._label(rec)
        if self.elaboration == "label":
            return key, label, {"value": key, "label": label}
        extras = {k: v for k, v in rec.items()
                  if isinstance(v, (str, int, float)) and not isinstance(v, bool) and len(str(v)) <= 80}
        compact = dict(list(extras.items())[: self.max_fields])
        text = " | ".join(str(v) for v in [label] + [f"{k}: {v}" for k, v in compact.items() if str(v) != label and str(v) != key] if v)
        return key, text[:300], compact

    def _rules(self, items):
        if self.elaboration != "full" or not self.sog:
            return []
        ents = {self._entity(i if isinstance(i, dict) else self.records.get(str(i), {})) for i in items}
        rules = []
        for e in self.sog.objects.entities:
            if e.id in ents:
                if e.description:
                    rules.append(f"{e.id}: {e.description}")
                rules.extend(e.rules)
        return rules

    def _instructions(self, question, rules, **data):
        if not rules and not data:
            return question
        return {"question": question, **data, **({"rules": rules} if rules else {})}

    # ---------- System-1 call ----------
    def _ask(self, state, questions, kind):
        from typesafe_sdk import Choice, Noul
        raw = {k: q for k, q in questions.items()}
        key = hashlib.sha256(json.dumps([self.model, state, raw], ensure_ascii=False, sort_keys=True, default=str)
                             .encode()).hexdigest()[:32]
        path = self.cache_dir / f"{key}.json"
        if self.use_cache and path.exists():
            out = json.loads(path.read_text(encoding="utf-8"))
            out["cached"] = True
        else:
            typed = {k: (Choice(instructions=q["instructions"], criteria=q["criteria"]) if q["type"] == "choice"
                         else Noul(instructions=q["instructions"])) for k, q in questions.items()}
            t0 = time.perf_counter()
            r = self.client.system_one(state, typed)
            latency = time.perf_counter() - t0
            answers = {}
            for k, a in r.answers.items():
                answers[k] = ({"probabilities": dict(a.probabilities)} if questions[k]["type"] == "choice"
                              else {"noul": a.noul})
            out = {"answers": answers, "tokens": r.usage.input_tokens, "latency": latency, "model": r.model,
                   "cached": False}
            if self.use_cache:
                path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        if self.ledger is not None:
            self.ledger.add("jev", tag=kind, latency=out["latency"], cached=out["cached"], input_tokens=out["tokens"],
                            questions=len(questions))
        return out

    def _state(self):
        return {"customer_request": self.request}

    # ---------- primitives ----------
    def choice(self, question, items, k=1, key_field=None):
        items = list(items or [])
        if not items:
            return [] if k != 1 else None
        if k != 1:
            return self._multi(question, items, kind="choice*")
        if len(items) > 255:                           # two-level: best of each chunk, then best of winners
            winners = [self.choice(question, items[i:i + 200], k=1, key_field=key_field) for i in range(0, len(items), 200)]
            return self.choice(question, [w for w in winners if w is not None], k=1, key_field=key_field)
        opts, by_key = {}, {}
        for it in items:
            key, text, _ = self._option(it, key_field)
            if key not in opts:
                opts[key], by_key[key] = text, it
        opts[NONE] = "以上都不是，或者无法确定"
        rules = self._rules(items)
        keys = list(opts)
        questions = {}
        for r in range(self.debias):                   # cyclic shifts of option order, averaged
            shift = keys[r * len(keys) // self.debias:] + keys[: r * len(keys) // self.debias]
            questions[f"q{r}"] = {"type": "choice", "instructions": self._instructions(question, rules),
                                  "criteria": {kk: opts[kk] for kk in shift}}
        out = self._ask(self._state(), questions, "choice")
        probs = {kk: sum(out["answers"][q]["probabilities"].get(kk, 0) for q in questions) / len(questions) for kk in keys}
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        top, p_top = ranked[0]
        if self.gate.conformal_tau is not None:
            pset = [kk for kk, p in probs.items() if 1 - p <= self.gate.conformal_tau]
            confident = pset == [top]
        else:
            confident = p_top >= self.gate.choice_min
        confident = confident and top != NONE
        gated = "act"
        if not confident and self.gate.enabled:
            if self.fallback:
                top = self.fallback("choice", question, opts, self.request)
                gated = "fallback"
            else:
                gated = "ungated"
        result = by_key.get(top)
        self.decisions.append(Decision("choice", question, opts, probs, top, gated, out["latency"], out["tokens"]))
        return result

    def _multi(self, question, items, kind):
        opts, by_key = {}, {}
        for it in items:
            key, text, compact = self._option(it)
            if key not in opts:
                opts[key], by_key[key] = (text, compact), it
        rules = self._rules(items)
        keys = list(opts)
        questions = {f"c{i}": {"type": "noul", "instructions": self._instructions(question, rules, candidate=opts[kk][1])}
                     for i, kk in enumerate(keys)}
        out = self._ask(self._state(), questions, kind)
        probs = {kk: out["answers"][f"c{i}"]["noul"] for i, kk in enumerate(keys)}
        keep = [kk for kk, p in probs.items() if p >= self.gate.noul_hi]
        unsure = [kk for kk, p in probs.items() if self.gate.noul_lo < p < self.gate.noul_hi]
        gated = "act"
        if unsure and self.gate.enabled:
            if self.fallback:
                keep += self.fallback("filter", question, {kk: opts[kk][0] for kk in unsure}, self.request)
                gated = "fallback"
            else:
                keep += [kk for kk in unsure if probs[kk] >= 0.5]
                gated = "ungated"
        elif not self.gate.enabled:
            keep = [kk for kk, p in probs.items() if p >= 0.5]
        result = [by_key[kk] for kk in keys if kk in set(keep)]
        self.decisions.append(Decision(kind, question, {kk: v[0] for kk, v in opts.items()}, probs,
                                       [kk for kk in keys if kk in set(keep)], gated, out["latency"], out["tokens"]))
        return result

    def filter(self, question, items):
        items = list(items or [])
        return self._multi(question, items, "filter") if items else []

    def noul(self, question, item=None):
        cand = None
        if item is not None:
            _, _, cand = self._option(item) if isinstance(item, dict) else (None, None, {"value": item})
        q = {"x": {"type": "noul", "instructions": self._instructions(question, self._rules([item] if item else []),
                                                                       **({"item": cand} if cand else {}))}}
        out = self._ask(self._state(), q, "noul")
        p = out["answers"]["x"]["noul"]
        gated = "act"
        if p >= self.gate.noul_hi:
            result = True
        elif p <= self.gate.noul_lo:
            result = False
        elif self.fallback and self.gate.enabled:
            result, gated = bool(self.fallback("noul", question, {"item": json.dumps(cand, ensure_ascii=False)}, self.request)), "fallback"
        else:
            result, gated = p >= 0.5, "ungated"
        self.decisions.append(Decision("noul", question, {"item": cand}, {"yes": p}, result, gated, out["latency"], out["tokens"]))
        return result
