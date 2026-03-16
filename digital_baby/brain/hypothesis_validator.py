"""Real-world hypothesis validation for digital_baby — v4.

Changes from v3
---------------
1. SIMULATION-FIRST VALIDATION (new primary source):
   Before any network call, checks the agent's own CausalRecord from
   InterventionCausalDiscovery.  If >= MIN_INTERNAL_EVIDENCE observations exist
   with clear directional confidence, returns SUPPORTED or CONTRADICTED
   immediately.  This is the fastest and most relevant signal for simulation-
   derived hypotheses (like velocity→acceleration).

2. CAUSAL AXIOM CACHE:
   Definitionally-true relationships (F=ma, Arrhenius, predator-prey) are
   pre-loaded and return SUPPORTED instantly with a large confidence boost.
   The agent should never waste experiments rediscovering Newton's second law.

3. SOFT INCONCLUSIVE DELTAS:
   Previously INCONCLUSIVE always returned confidence_delta=0.0, meaning
   hypotheses stagnated forever at their initial confidence.  Now:
   - Co-occurrence found but no causal language: +0.02 (soft positive)
   - No co-occurrence in either Wikipedia page:  -0.01 (soft negative)
   These small signals compound over many validation cycles.

4. HYPOTHESIS LIFECYCLE STATUS:
   HypothesisValidator now reads and writes a 'status' field on hypotheses:
   CANDIDATE → ACTIVE → CONFIRMED / REFUTED / SUSPENDED / ABANDONED
   Hypotheses with status SUSPENDED or ABANDONED are skipped, preventing
   the validator from wasting cycles on unresolvable relationships.

5. VALIDATION RESULT AGGREGATION:
   Results from simulation + Wikipedia + Wikidata are merged with source-
   specific weights, so all three sources contribute to the final delta.

All v3 public API is preserved.
"""

from __future__ import annotations

import json as _json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Causal / contradiction language patterns (unchanged from v3)
# ---------------------------------------------------------------------------

_CAUSAL_PATTERNS = re.compile(
    r"\b(causes?|affects?|increases?|promotes?|enables?|triggers?|"
    r"leads?\s+to|results?\s+in|drives?|produces?|generates?|"
    r"stimulates?|accelerates?|inhibits?|reduces?|decreases?)\b",
    re.I,
)
_CONTRADICTION_PATTERNS = re.compile(
    r"\b(does\s+not|cannot|prevents?|inhibits?|blocks?|stops?|"
    r"unrelated\s+to|has\s+no\s+effect|independent\s+of)\b",
    re.I,
)
_PROXIMITY_WINDOW = 120

# ---------------------------------------------------------------------------
# Hypothesis lifecycle statuses
# ---------------------------------------------------------------------------

class HypothesisStatus(str, Enum):
    CANDIDATE  = "candidate"   # newly generated, not yet tested
    ACTIVE     = "active"      # eligible for testing
    CONFIRMED  = "confirmed"   # confidence >= 0.70 with evidence >= 5
    REFUTED    = "refuted"     # confidence <= 0.25 with evidence >= 5
    SUSPENDED  = "suspended"   # ambiguous after 10+ experiments (defer)
    ABANDONED  = "abandoned"   # ambiguous after 20+ experiments (archive)

# Confidence threshold below which a hypothesis is considered REFUTED
REFUTED_CONFIDENCE:   float = 0.25
# Confidence threshold above which a hypothesis is CONFIRMED
CONFIRMED_CONFIDENCE: float = 0.70
# Evidence required to move to CONFIRMED/REFUTED
MIN_DECISIVE_EVIDENCE: int  = 5
# Evidence required before suspension
MIN_SUSPEND_EVIDENCE:  int  = 10
# Evidence required before abandonment
MIN_ABANDON_EVIDENCE:  int  = 20
# Confidence band considered "ambiguous" (neither confirmed nor refuted)
AMBIGUOUS_LO: float = 0.35
AMBIGUOUS_HI: float = 0.55

# ---------------------------------------------------------------------------
# Minimum internal evidence for simulation-first validation
# ---------------------------------------------------------------------------
MIN_INTERNAL_EVIDENCE: int = 3

# ---------------------------------------------------------------------------
# Causal axioms: (concept_a, concept_b) -> (verdict, delta, description)
# These bypass all external API calls.
# ---------------------------------------------------------------------------
_CAUSAL_AXIOMS_VALIDATION: Dict[Tuple[str, str], Tuple[str, float, str]] = {
    # physics
    ("force",       "acceleration"):    ("supported", +0.30, "Newton_second_law_F=ma"),
    ("mass",        "kinetic_energy"):  ("supported", +0.25, "KE=0.5mv2"),
    ("velocity",    "kinetic_energy"):  ("supported", +0.25, "KE=0.5mv2"),
    ("heat",        "temperature"):     ("supported", +0.25, "thermodynamics"),
    ("friction",    "kinetic_energy"):  ("supported", +0.25, "friction_dissipates_energy"),
    ("friction",    "acceleration"):    ("supported", +0.20, "net_force_reduced_by_friction"),
    # chemistry
    ("temperature", "reaction_rate"):   ("supported", +0.25, "Arrhenius_equation"),
    ("catalyst",    "reaction_rate"):   ("supported", +0.25, "catalysis"),
    ("pH",          "reaction_rate"):   ("supported", +0.15, "acid_base_kinetics"),
    # ecology
    ("wolf",        "deer"):            ("supported", +0.25, "predator_prey_dynamics"),
    ("deer",        "grass"):           ("supported", +0.25, "herbivory"),
    ("wolf",        "grass"):           ("supported", +0.20, "trophic_cascade"),
    # biology
    ("pathogens",   "immune_response"): ("supported", +0.25, "immune_activation"),
    ("antibodies",  "pathogens"):       ("supported", +0.25, "antibody_neutralisation"),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    SUPPORTED    = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    SKIPPED      = "skipped"


@dataclass
class ValidationResult:
    hypothesis_rule:  str
    concept_a:        str
    concept_b:        str
    verdict:          Verdict
    confidence_delta: float
    evidence_text:    str
    source:           str      # "axiom" | "simulation" | "wikipedia" | "wikidata" | "none"
    timestamp:        float = 0.0
    status_update:    Optional[str] = None   # v4: new lifecycle status if changed

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 8.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "digital_baby/4.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("[validator] http_failed url=%s error=%s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Concept variants
# ---------------------------------------------------------------------------

def _concept_variants(concept: str) -> List[str]:
    c = concept.replace("_", " ").lower().strip()
    variants = [c]
    if c.endswith("s") and len(c) > 3:
        variants.append(c[:-1])
        variants.append(c[:-2])
    else:
        variants.append(c + "s")
        variants.append(c + "es")
    return variants


def _find_in_summary(needle_variants: List[str], summary: str) -> Optional[int]:
    for v in needle_variants:
        idx = summary.find(v)
        if idx != -1:
            return idx
    return None


def _score_window(
    summary: str, idx: int, match_len: int, label_a: str, label_b: str
) -> Tuple[Verdict, float, str]:
    window_start = max(0, idx - _PROXIMITY_WINDOW)
    window_end   = min(len(summary), idx + match_len + _PROXIMITY_WINDOW)
    window = summary[window_start:window_end]

    if _CONTRADICTION_PATTERNS.search(window):
        return Verdict.CONTRADICTED, -0.15, f"contradiction_near_{label_b}_in_{label_a}_wiki"
    if _CAUSAL_PATTERNS.search(window):
        return Verdict.SUPPORTED, +0.12, f"causal_near_{label_b}_in_{label_a}_wiki"
    # v4: soft positive for co-occurrence without causal language (was +0.05 SUPPORTED)
    return Verdict.INCONCLUSIVE, +0.03, f"soft_cooccurrence_{label_a}_{label_b}_wiki"


def _wikipedia_cooccurrence(
    concept_a: str,
    concept_b: str,
    timeout: float = 8.0,
) -> Tuple[Verdict, float, str]:
    # ── Try tools bridge first (uses wikipedia tool) ─────────────────────
    try:
        from digital_baby.tools_bridge.wikipedia import fetch_summary as _bfetch
        summary_a = (_bfetch(concept_a, full=True) or "").lower()
        summary_b = (_bfetch(concept_b, full=True) or "").lower()
        if summary_a and summary_b:
            a_variants = _concept_variants(concept_a)
            b_variants = _concept_variants(concept_b)
            idx_ab = _find_in_summary(b_variants, summary_a)
            idx_ba = _find_in_summary(a_variants, summary_b)
            if idx_ab is not None:
                matched = next(v for v in b_variants if summary_a.find(v) == idx_ab)
                return _score_window(summary_a, idx_ab, len(matched), concept_a, concept_b)
            if idx_ba is not None:
                matched = next(v for v in a_variants if summary_b.find(v) == idx_ba)
                return _score_window(summary_b, idx_ba, len(matched), concept_b, concept_a)
            return Verdict.INCONCLUSIVE, -0.01, f"{concept_b}_not_found_in_either_wiki_page(bridge)"
    except Exception:
        pass  # fall through to raw urllib

    # ── Raw urllib fallback (original implementation) ────────────────────
    a_variants = _concept_variants(concept_a)
    b_variants = _concept_variants(concept_b)

    def _fetch_summary(concept: str) -> Optional[str]:
        title = concept.replace("_", " ")
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title)
        )
        data = _http_get(url, timeout)
        if not data:
            return None
        if "disambiguation" in data.get("type", "").lower():
            return None
        return (data.get("extract") or "").lower()

    summary_a = _fetch_summary(concept_a)
    if summary_a is not None:
        idx = _find_in_summary(b_variants, summary_a)
        if idx is not None:
            matched = next(v for v in b_variants if summary_a.find(v) == idx)
            return _score_window(summary_a, idx, len(matched), concept_a, concept_b)

    summary_b = _fetch_summary(concept_b)
    if summary_b is not None:
        idx = _find_in_summary(a_variants, summary_b)
        if idx is not None:
            matched = next(v for v in a_variants if summary_b.find(v) == idx)
            return _score_window(summary_b, idx, len(matched), concept_b, concept_a)

    if summary_a is None and summary_b is None:
        return Verdict.SKIPPED, 0.0, "no_wikipedia_response"

    return Verdict.INCONCLUSIVE, -0.01, f"{concept_b}_not_found_in_either_wiki_page"


def _wikidata_direct_link(
    concept_a: str,
    concept_b: str,
    timeout: float = 8.0,
) -> Tuple[Verdict, float, str]:
    def get_qid(label: str) -> Optional[str]:
        url = (
            "https://www.wikidata.org/w/api.php"
            "?action=wbsearchentities"
            f"&search={urllib.parse.quote(label.replace('_', ' '))}"
            "&language=en&limit=3&format=json"
        )
        data = _http_get(url, timeout)
        if not data:
            return None
        items = data.get("search", [])
        return items[0].get("id") if items else None

    qid_a = get_qid(concept_a)
    qid_b = get_qid(concept_b)
    if not qid_a or not qid_b:
        return Verdict.SKIPPED, 0.0, "qid_resolution_failed"

    sparql = f"ASK {{ wd:{qid_a} ?prop wd:{qid_b} }}"
    url = (
        "https://query.wikidata.org/sparql?format=json&query="
        + urllib.parse.quote(sparql)
    )
    data = _http_get(url, timeout)
    if not data:
        return Verdict.SKIPPED, 0.0, "sparql_failed"

    if data.get("boolean", False):
        return (
            Verdict.SUPPORTED,
            +0.18,
            f"wikidata_direct_link_{qid_a}_{qid_b}",
        )

    return Verdict.INCONCLUSIVE, -0.01, f"no_wikidata_link_{qid_a}_{qid_b}"


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

class HypothesisValidator:
    """Validates agent hypotheses — v4.

    Validation order:
    1. Causal axiom cache        (instant, highest confidence boost)
    2. Internal simulation data  (agent's own CausalRecords)
    3. Wikipedia co-occurrence   (external natural language)
    4. Wikidata structural link  (external ontology)

    Hypotheses with status SUSPENDED or ABANDONED are skipped entirely.

    Parameters
    ----------
    validation_interval : ticks between validation runs (default 75)
    batch_size          : hypotheses validated per run (default 3)
    request_timeout     : seconds per network call (default 8.0)
    enable_wikipedia    : toggle Wikipedia source
    enable_wikidata     : toggle Wikidata source
    enable_simulation   : toggle internal simulation evidence (default True)
    enable_axioms       : toggle causal axiom cache (default True)
    """

    def __init__(
        self,
        validation_interval: int = 75,
        batch_size: int = 3,
        request_timeout: float = 8.0,
        enable_wikipedia: bool = True,
        enable_wikidata: bool = True,
        enable_simulation: bool = True,
        enable_axioms: bool = True,
    ) -> None:
        self.validation_interval = max(1, validation_interval)
        self.batch_size          = max(1, batch_size)
        self.request_timeout     = request_timeout
        self.enable_wikipedia    = enable_wikipedia
        self.enable_wikidata     = enable_wikidata
        self.enable_simulation   = enable_simulation
        self.enable_axioms       = enable_axioms

        self._last_run_tick: int = 0
        self._validated: Dict[str, ValidationResult] = {}

        # Track how many times each hypothesis has been tested (for lifecycle)
        self._test_count: Dict[str, int] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def maybe_validate(
        self,
        tick: int,
        hypotheses: List,
        memory,
        causal_records: Optional[Dict] = None,   # v4: pass CausalRecord dict
    ) -> List[ValidationResult]:
        if tick - self._last_run_tick < self.validation_interval:
            return []
        self._last_run_tick = tick
        return self._run_validation(hypotheses, memory, causal_records or {})

    def get_lifecycle_status(self, hypothesis) -> str:
        """v4: Compute current lifecycle status for a hypothesis."""
        rule = hypothesis.rule
        conf = float(getattr(hypothesis, "confidence", 0.5))
        supp = int(getattr(hypothesis, "supporting_evidence", 0))
        cont = int(getattr(hypothesis, "contradicting_evidence", 0))
        total_evidence = supp + cont

        # Check for confirmed/refuted with sufficient evidence
        if total_evidence >= MIN_DECISIVE_EVIDENCE:
            if conf >= CONFIRMED_CONFIDENCE:
                return HypothesisStatus.CONFIRMED
            if conf <= REFUTED_CONFIDENCE:
                return HypothesisStatus.REFUTED

        # Check for suspension/abandonment based on test count + ambiguity
        tests = self._test_count.get(rule, 0)
        if AMBIGUOUS_LO <= conf <= AMBIGUOUS_HI:
            if total_evidence >= MIN_ABANDON_EVIDENCE and tests >= 20:
                return HypothesisStatus.ABANDONED
            if total_evidence >= MIN_SUSPEND_EVIDENCE and tests >= 10:
                return HypothesisStatus.SUSPENDED

        return HypothesisStatus.ACTIVE

    # ── Internal ────────────────────────────────────────────────────────────

    def _pick_candidates(self, hypotheses: List) -> List:
        """Select hypotheses eligible for validation.

        v4: Filters out SUSPENDED and ABANDONED hypotheses.
        Prioritises those closest to the uncertainty midpoint (0.5 confidence).
        """
        eligible = []
        for h in hypotheses:
            rule   = h.rule
            status = self.get_lifecycle_status(h)
            if status in (HypothesisStatus.SUSPENDED, HypothesisStatus.ABANDONED):
                logger.debug("[validator] skipping %s status=%s", rule[:40], status)
                continue
            if rule in self._validated:
                # Allow re-validation every 20 tests to catch updated evidence
                tests = self._test_count.get(rule, 0)
                if tests > 0 and tests % 20 != 0:
                    continue
            eligible.append(h)

        # Sort: most uncertain first (closest to 0.5)
        eligible.sort(key=lambda h: abs(float(getattr(h, "confidence", 0.5)) - 0.5))
        return eligible[: self.batch_size]

    def _extract_concepts(self, hypothesis) -> Tuple[Optional[str], Optional[str]]:
        concepts = getattr(hypothesis, "concepts", [])
        if len(concepts) >= 2:
            a = concepts[0].replace(" ", "_").lower()
            b = concepts[1].replace(" ", "_").lower()
            if len(a) > 2 and len(b) > 2:
                return a, b

        tokens = hypothesis.rule.lower().split()
        stop = {"if", "then", "will", "may", "the", "a", "an",
                "increases", "decreases", "changes", "affects"}
        meaningful = [t for t in tokens if len(t) > 2 and t not in stop]
        if len(meaningful) >= 2:
            return meaningful[0], meaningful[-1]

        return None, None

    def _validate_from_axiom(
        self, concept_a: str, concept_b: str
    ) -> Optional[Tuple[Verdict, float, str]]:
        """v4: Check causal axiom cache first."""
        if not self.enable_axioms:
            return None
        key = (concept_a, concept_b)
        result = _CAUSAL_AXIOMS_VALIDATION.get(key)
        if result:
            verdict_str, delta, text = result
            verdict = Verdict.SUPPORTED if verdict_str == "supported" else Verdict.CONTRADICTED
            return verdict, delta, text
        # Also check reverse direction (B→A might be contradicted by A→B axiom)
        rev = (concept_b, concept_a)
        if rev in _CAUSAL_AXIOMS_VALIDATION:
            # Reverse direction exists — this direction is not an axiom (not contradicted, just unknown)
            pass
        return None

    def _validate_from_simulation(
        self, concept_a: str, concept_b: str, causal_records: Dict
    ) -> Optional[Tuple[Verdict, float, str]]:
        """v4: Check agent's own CausalRecords from simulation experiments."""
        if not self.enable_simulation:
            return None

        key    = (concept_a, concept_b)
        record = causal_records.get(key)
        if record is None:
            # Try reverse direction
            rev = (concept_b, concept_a)
            record = causal_records.get(rev)
            if record is None:
                return None

        if record.evidence_count < MIN_INTERNAL_EVIDENCE:
            return None   # too few experiments, fall through to external sources

        conf = record.confidence
        ev   = record.evidence_count

        if conf >= 0.65:
            # Scale delta by evidence: more experiments = higher boost
            delta = min(0.20, 0.08 + 0.012 * min(10, ev))
            return (
                Verdict.SUPPORTED,
                delta,
                f"simulation_confirmed n={ev} conf={conf:.2f}",
            )
        if conf <= 0.35:
            return (
                Verdict.CONTRADICTED,
                -0.10,
                f"simulation_refuted n={ev} conf={conf:.2f}",
            )

        # Ambiguous — return a small soft positive for having tried
        return (
            Verdict.INCONCLUSIVE,
            +0.01,
            f"simulation_ambiguous n={ev} conf={conf:.2f}",
        )

    def _validate_one(
        self, hypothesis, causal_records: Dict
    ) -> ValidationResult:
        concept_a, concept_b = self._extract_concepts(hypothesis)

        if not concept_a or not concept_b:
            return ValidationResult(
                hypothesis_rule=hypothesis.rule,
                concept_a=concept_a or "unknown",
                concept_b=concept_b or "unknown",
                verdict=Verdict.SKIPPED,
                confidence_delta=0.0,
                evidence_text="concept_extraction_failed",
                source="none",
            )

        # Track test count for lifecycle
        self._test_count[hypothesis.rule] = self._test_count.get(hypothesis.rule, 0) + 1

        # ── Layer 1: Causal axiom cache ──────────────────────────────────
        axiom_result = self._validate_from_axiom(concept_a, concept_b)
        if axiom_result is not None:
            verdict, delta, evidence = axiom_result
            result = ValidationResult(
                hypothesis_rule=hypothesis.rule,
                concept_a=concept_a, concept_b=concept_b,
                verdict=verdict, confidence_delta=delta,
                evidence_text=evidence, source="axiom",
            )
            logger.info(
                "[validator] axiom_hit rule=%r verdict=%s delta=%+.2f",
                hypothesis.rule[:60], verdict.value, delta,
            )
            return result

        # ── Layer 2: Simulation evidence ─────────────────────────────────
        sim_result = self._validate_from_simulation(concept_a, concept_b, causal_records)
        if sim_result is not None:
            sim_verdict, sim_delta, sim_evidence = sim_result
            if sim_verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED):
                result = ValidationResult(
                    hypothesis_rule=hypothesis.rule,
                    concept_a=concept_a, concept_b=concept_b,
                    verdict=sim_verdict, confidence_delta=sim_delta,
                    evidence_text=sim_evidence, source="simulation",
                )
                logger.info(
                    "[validator] simulation_validated rule=%r verdict=%s delta=%+.2f",
                    hypothesis.rule[:60], sim_verdict.value, sim_delta,
                )
                return result
        else:
            sim_delta    = 0.0
            sim_evidence = "no_internal_evidence"

        # ── Layers 3 & 4: External sources ───────────────────────────────
        best_verdict  = Verdict.INCONCLUSIVE
        best_delta    = sim_delta   # carry forward any soft simulation signal
        best_evidence = sim_evidence
        best_source   = "simulation" if sim_delta != 0.0 else "none"

        if self.enable_wikipedia:
            verdict, delta, evidence = _wikipedia_cooccurrence(
                concept_a, concept_b, self.request_timeout
            )
            if verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED):
                best_verdict  = verdict
                best_delta    = delta + sim_delta   # v4: combine with simulation signal
                best_evidence = evidence
                best_source   = "wikipedia"
            elif delta != 0.0:
                # Soft INCONCLUSIVE — accumulate signal
                best_delta    += delta
                best_evidence = evidence
                best_source   = "wikipedia+soft"

        if self.enable_wikidata and best_verdict == Verdict.INCONCLUSIVE:
            verdict, delta, evidence = _wikidata_direct_link(
                concept_a, concept_b, self.request_timeout
            )
            if verdict != Verdict.SKIPPED:
                best_verdict  = verdict
                best_delta    += delta
                best_evidence = evidence
                best_source   = "wikidata"

        # Determine lifecycle status update
        new_conf  = float(getattr(hypothesis, "confidence", 0.5)) + best_delta
        status_update = None
        total_ev = int(getattr(hypothesis, "supporting_evidence", 0)) + int(getattr(hypothesis, "contradicting_evidence", 0))
        if total_ev >= MIN_DECISIVE_EVIDENCE:
            if new_conf >= CONFIRMED_CONFIDENCE:
                status_update = HypothesisStatus.CONFIRMED
            elif new_conf <= REFUTED_CONFIDENCE:
                status_update = HypothesisStatus.REFUTED

        result = ValidationResult(
            hypothesis_rule=hypothesis.rule,
            concept_a=concept_a, concept_b=concept_b,
            verdict=best_verdict, confidence_delta=best_delta,
            evidence_text=best_evidence, source=best_source,
            status_update=status_update,
        )

        logger.info(
            "[validator] rule=%r verdict=%s delta=%+.2f source=%s",
            hypothesis.rule[:60], best_verdict.value, best_delta, best_source,
        )
        return result

    def _run_validation(
        self,
        hypotheses: List,
        memory,
        causal_records: Dict,
    ) -> List[ValidationResult]:
        candidates = self._pick_candidates(hypotheses)
        if not candidates:
            return []

        results: List[ValidationResult] = []
        for hypothesis in candidates:
            result = self._validate_one(hypothesis, causal_records)
            self._validated[hypothesis.rule] = result
            results.append(result)

            if result.verdict == Verdict.SUPPORTED:
                memory.update_hypothesis_evidence(
                    hypothesis.rule, supported=1, contradicted=0,
                )
            elif result.verdict == Verdict.CONTRADICTED:
                memory.update_hypothesis_evidence(
                    hypothesis.rule, supported=0, contradicted=1,
                )
            elif result.confidence_delta != 0.0:
                # v4: apply soft deltas even for INCONCLUSIVE results
                if result.confidence_delta > 0:
                    memory.update_hypothesis_evidence(
                        hypothesis.rule, supported=1, contradicted=0,
                    )
                else:
                    memory.update_hypothesis_evidence(
                        hypothesis.rule, supported=0, contradicted=1,
                    )

        return results