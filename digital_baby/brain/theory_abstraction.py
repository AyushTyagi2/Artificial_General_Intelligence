"""Theory Abstraction — v1.

Detects recurring mathematical structures in discovered laws and lifts them
into reusable, domain-independent templates.

Design
------
When law_discovery produces a DiscoveredLaw, this module:

1. Classifies the equation family (linear, exponential, logarithmic, …).
2. Maps variable names to abstract semantic roles (barrier, rate, resource, …).
3. Creates or updates a TheoryTemplate that groups all laws sharing the same
   structural form.
4. Exposes templates for reuse — e.g. so the ResearchAgent can predict that a
   newly discovered variable pair might follow the same pattern as an existing
   template.

Templates are stored in-memory as a dict keyed by template name.  No external
dependencies are required.

Integration point
-----------------
Call ``absorb_law(law)`` from event_loop.py immediately after a novel law
passes the LawNoveltyGate.  Example (in the law-discovery block):

    if law_eval.is_novel:
        self.knowledge_graph.ingest_law(law)
        self.theory_abstraction.absorb_law(law)   # <-- add this line
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic role vocabulary
# ---------------------------------------------------------------------------

# Maps variable substrings → abstract role name.
# Checked in the order listed; first match wins.
_ROLE_HINTS: List[tuple[str, str]] = [
    ("activation_energy", "barrier"),
    ("energy",            "barrier"),
    ("stress",            "barrier"),
    ("threshold",         "barrier"),
    ("cost",              "barrier"),
    ("resistance",        "barrier"),
    ("temperature",       "rate"),
    ("reaction_rate",     "rate"),
    ("growth",            "rate"),
    ("speed",             "rate"),
    ("velocity",          "rate"),
    ("population",        "population"),
    ("concentration",     "concentration"),
    ("pressure",          "pressure"),
    ("force",             "force"),
    ("mass",              "mass"),
    ("resource",          "resource"),
    ("nutrient",          "resource"),
    ("food",              "resource"),
    ("light",             "stimulus"),
    ("signal",            "stimulus"),
    ("stimulus",          "stimulus"),
    ("time",              "time"),
    ("distance",          "distance"),
    ("entropy",           "entropy"),
]


def _semantic_role(variable_name: str) -> str:
    """Map a concrete variable name to its abstract semantic role."""
    name = variable_name.lower().replace("-", "_")
    for hint, role in _ROLE_HINTS:
        if hint in name:
            return role
    # Fallback: use the variable name with underscores stripped
    return name.replace("_", " ").split()[0]


# ---------------------------------------------------------------------------
# Equation family classification
# ---------------------------------------------------------------------------

# (pattern, template_name, abstract_equation)
_EQUATION_PATTERNS: List[tuple[str, str, str]] = [
    # Arrhenius / barrier process  y = k * exp(-E/x)
    (r"exp\s*\(\s*-",          "barrier_process",    "rate ~ exp(-barrier / x)"),
    # Plain exponential           y = k * exp(b*x)
    (r"exp\s*\(",              "exponential_growth",  "y ~ exp(k * x)"),
    # Logarithmic                 y = k * log(x)
    (r"log\s*\(",              "logarithmic_return",  "y ~ log(x)"),
    # Power / square root         y = k * x^0.5
    (r"sqrt\s*\(\s*x\s*\)|x\s*\*\*\s*0\.5",
                               "sqrt_growth",         "y ~ sqrt(x)"),
    # Quadratic                   y = k * x^2
    (r"x\s*\*\*\s*2",         "quadratic_growth",    "y ~ x^2"),
    # Inverse                     y = k / x
    (r"k\s*/\s*x|/\s*x",     "inverse_decay",       "y ~ 1/x"),
    # Linear (last — most general)
    (r"k\s*\*\s*x|=\s*[+-]?\s*\d+\.\d+\s*\*\s*x",
                               "linear_proportional", "y ~ k * x"),
]


def _classify_equation(equation_str: str) -> tuple[str, str]:
    """Return (template_name, abstract_equation) for the given equation string."""
    s = equation_str.lower()
    for pattern, name, abstract in _EQUATION_PATTERNS:
        if re.search(pattern, s):
            return name, abstract
    return "unknown_structure", "y ~ f(x)"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SupportingLaw:
    cause: str
    effect: str
    equation_str: str
    r_squared: float
    cause_role: str
    effect_role: str


@dataclass
class TheoryTemplate:
    """A reusable mathematical pattern shared by one or more discovered laws."""
    name: str
    abstract_equation: str
    supporting_laws: List[SupportingLaw] = field(default_factory=list)

    # Known variable roles that map to the abstract roles in this template.
    # e.g. {"barrier": {"activation_energy", "stress_threshold"}}
    role_instances: Dict[str, set] = field(default_factory=dict)

    @property
    def n_laws(self) -> int:
        return len(self.supporting_laws)

    def add_role_instance(self, role: str, variable: str) -> None:
        self.role_instances.setdefault(role, set()).add(variable)

    def __str__(self) -> str:
        return (
            f"TheoryTemplate(name={self.name!r}, "
            f"equation={self.abstract_equation!r}, "
            f"n_laws={self.n_laws})"
        )


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class TheoryAbstraction:
    """Absorbs discovered laws and builds abstract theoretical templates."""

    def __init__(self) -> None:
        # template_name → TheoryTemplate
        self._templates: Dict[str, TheoryTemplate] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def absorb_law(self, law) -> Optional[TheoryTemplate]:
        """Classify *law* and add it to the matching TheoryTemplate.

        Parameters
        ----------
        law : DiscoveredLaw  (from brain.law_discovery)
            Must have attributes: cause, effect, equation_str, r_squared.

        Returns
        -------
        TheoryTemplate that was updated, or None if the law is unclassifiable.
        """
        template_name, abstract_eq = _classify_equation(law.equation_str)

        cause_role  = _semantic_role(law.cause)
        effect_role = _semantic_role(law.effect)

        supporting = SupportingLaw(
            cause=law.cause,
            effect=law.effect,
            equation_str=law.equation_str,
            r_squared=law.r_squared,
            cause_role=cause_role,
            effect_role=effect_role,
        )

        if template_name not in self._templates:
            tmpl = TheoryTemplate(name=template_name, abstract_equation=abstract_eq)
            self._templates[template_name] = tmpl
            logger.info(
                "[theory_abstraction] new_template name=%r equation=%r",
                template_name, abstract_eq,
            )
        else:
            tmpl = self._templates[template_name]

        # Avoid duplicate laws
        existing_pairs = {(s.cause, s.effect) for s in tmpl.supporting_laws}
        if (law.cause, law.effect) not in existing_pairs:
            tmpl.supporting_laws.append(supporting)
            tmpl.add_role_instance(cause_role,  law.cause)
            tmpl.add_role_instance(effect_role, law.effect)
            logger.info(
                "[theory_abstraction] law_abstracted cause=%r effect=%r "
                "template=%r cause_role=%r effect_role=%r r2=%.3f",
                law.cause, law.effect, template_name,
                cause_role, effect_role, law.r_squared,
            )

        return tmpl

    def get_templates(self) -> List[TheoryTemplate]:
        """Return all templates with at least one supporting law."""
        return [t for t in self._templates.values() if t.n_laws > 0]

    def predict_template_for(self, cause: str, effect: str) -> Optional[TheoryTemplate]:
        """Return the best-matching template for a new variable pair, or None.

        Heuristic: look for any template whose known role instances overlap
        with the semantic roles of (cause, effect).
        """
        c_role = _semantic_role(cause)
        e_role = _semantic_role(effect)

        best: Optional[TheoryTemplate] = None
        best_score = 0

        for tmpl in self._templates.values():
            score = 0
            if c_role in tmpl.role_instances:
                score += 2
            if e_role in tmpl.role_instances:
                score += 2
            # Exact variable match is strongest
            for sl in tmpl.supporting_laws:
                if sl.cause == cause or sl.effect == effect:
                    score += 3
            if score > best_score:
                best_score = score
                best = tmpl

        return best if best_score >= 2 else None

    def summary(self) -> str:
        lines = [f"TheoryAbstraction: {len(self._templates)} template(s)"]
        for tmpl in sorted(self._templates.values(), key=lambda t: -t.n_laws):
            lines.append(
                f"  {tmpl.name:25s}  {tmpl.abstract_equation:35s}  "
                f"laws={tmpl.n_laws}"
            )
        return "\n".join(lines)