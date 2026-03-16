"""Screen Tracker — perception module for digital_baby.

Converts visual information from the screen into structured causal knowledge,
giving the system a second learning channel alongside internal simulation.

Pipeline
--------
capture_screen()
  → extract_text()      (Tesseract OCR or direct PIL grab)
  → extract_relations() (regex over causal trigger verbs)
  → align_nodes()       (match to existing KG nodes when possible)
  → ingest_perception() (episodic store + low-confidence KG edge)

Design constraints
------------------
* Never injects raw text into the knowledge graph.
* All perception edges start at confidence <= PERCEPTION_CONFIDENCE_MAX.
* Edges are promoted only after multi-observation support or experimental
  confirmation (handled by the existing confidence-update machinery).
* Runs asynchronously — does not block the event loop.
* Gracefully degrades: if no display is available, operate in text-feed mode
  only (caller passes text directly via ingest_text()).

Integration
-----------
In BabyEventLoop.__init__:
    from digital_baby.brain.perception import ScreenTracker
    self.screen_tracker = ScreenTracker(knowledge_graph=self.knowledge_graph)
    self.screen_tracker.start()

In the tick body (after ingestion):
    perc_edges = self.screen_tracker.drain_pending()
    for triple in perc_edges:
        self.knowledge_graph.add_or_update_edge(
            triple.subject, triple.object, triple.relation,
            confidence=triple.confidence,
            provenance="screen",
            current_tick=tick,
        )
        self.ingestion.suggest_concepts([triple.subject, triple.object])
"""

from __future__ import annotations

import io
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state file (written by dashboard, read by ScreenTracker)
# ---------------------------------------------------------------------------

def _get_screen_state_path() -> "Path":
    """Resolve the shared screen_capture_state.json path."""
    import os
    env = os.environ.get("DIGITAL_BABY_SCREEN_STATE", "")
    if env:
        return Path(env)
    # Walk up from this file to find the world/ directory
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "digital_baby" / "world" / "screen_capture_state.json"
        if candidate.parent.exists():
            return candidate
    return Path("digital_baby/world/screen_capture_state.json")

_SCREEN_STATE_PATH = _get_screen_state_path()
_STATE_POLL_INTERVAL_S: float = 2.0   # how often to re-read the file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAPTURE_INTERVAL_S:        float = 5.0    # seconds between screen captures
PERCEPTION_CONFIDENCE_MAX: float = 0.38   # ceiling for raw screen observations
MIN_ENTITY_LENGTH:         int   = 3      # shortest valid concept name
MAX_TRIPLES_PER_FRAME:     int   = 12     # cap to avoid bulk-injection spikes
DUPLICATE_WINDOW:          int   = 60     # seconds before re-emitting same triple
EPISODIC_LOG_MAXLEN:       int   = 200    # max perception events kept in memory

# Causal trigger verbs and their canonical relation mapping
_CAUSAL_VERBS: Dict[str, Tuple[str, str]] = {
    # verb              relation                direction
    "increases":    ("positive_affects",  "positive"),
    "improves":     ("positive_affects",  "positive"),
    "promotes":     ("positive_affects",  "positive"),
    "enhances":     ("positive_affects",  "positive"),
    "activates":    ("positive_affects",  "positive"),
    "stimulates":   ("positive_affects",  "positive"),
    "boosts":       ("positive_affects",  "positive"),
    "accelerates":  ("positive_affects",  "positive"),
    "decreases":    ("negative_affects",  "negative"),
    "reduces":      ("negative_affects",  "negative"),
    "inhibits":     ("negative_affects",  "negative"),
    "suppresses":   ("negative_affects",  "negative"),
    "slows":        ("negative_affects",  "negative"),
    "impairs":      ("negative_affects",  "negative"),
    "limits":       ("negative_affects",  "negative"),
    "blocks":       ("negative_affects",  "negative"),
    "causes":       ("affects",           "mixed"),
    "affects":      ("affects",           "mixed"),
    "influences":   ("affects",           "mixed"),
    "controls":     ("affects",           "mixed"),
    "regulates":    ("affects",           "mixed"),
    "modulates":    ("affects",           "mixed"),
    "drives":       ("affects",           "mixed"),
    "triggers":     ("affects",           "mixed"),
}

# Build combined regex: "(subject) (verb) (object)"
# Also handles "X improves Y's Z" → X improves Z
_VERB_ALTERNATION = "|".join(sorted(_CAUSAL_VERBS.keys(), key=len, reverse=True))
_RELATION_RE = re.compile(
    r"(?:^|(?<=\.\s)|(?<=\n)|\b)"
    r"((?:[A-Za-z][a-z_]*(?:[\s_][a-z]+){0,3}))"   # subject (1-4 words)
    r"\s+"
    r"(?:can\s+|may\s+|will\s+|also\s+|directly\s+|indirectly\s+)?"
    r"(" + _VERB_ALTERNATION + r")"                  # verb
    r"\s+"
    r"(?:the\s+|a\s+|an\s+|its\s+)?"
    r"((?:[A-Za-z][a-z_]*(?:[\s_][a-z]+){0,3}))",   # object (1-4 words)
    re.IGNORECASE,
)

# Stop words to drop from entity detection
_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "it", "its", "this", "that", "which",
    "also", "such", "more", "most", "many", "some", "other", "new", "high",
    "low", "large", "small", "major", "key", "important", "significant",
    "recent", "overall", "general", "specific", "various", "different",
    "studies", "research", "study", "results", "evidence", "data", "effect",
    "effects", "level", "levels", "rate", "rates", "role", "roles",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerceptionTriple:
    """One structured relation extracted from screen text."""
    subject:    str
    relation:   str
    object:     str
    confidence: float
    direction:  str       # "positive" | "negative" | "mixed"
    source_text: str      # snippet the triple was derived from
    timestamp:  float = field(default_factory=time.time)
    tick:       int   = 0


@dataclass
class PerceptionEvent:
    """One screen-capture episode stored in episodic memory."""
    timestamp:  float
    source:     str          # "screen" | "text_feed"
    raw_text:   str
    entities:   List[str]
    triples:    List[PerceptionTriple]
    frame_hash: str          # dedup key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_concept(raw: str) -> Optional[str]:
    """Normalise a raw phrase to a canonical concept name."""
    c = raw.strip().lower().replace(" ", "_").replace("-", "_")
    c = re.sub(r"[^a-z0-9_]", "", c)
    c = re.sub(r"_+", "_", c).strip("_")
    if len(c) < MIN_ENTITY_LENGTH:
        return None
    if c in _STOP_WORDS:
        return None
    if c.replace("_", "").isdigit():
        return None
    return c


def _extract_entities_from_text(text: str) -> List[str]:
    """Extract candidate scientific entities from free text."""
    # Match noun phrases: 1-3 capitalised or lower-case scientific words
    pattern = re.compile(
        r"\b([A-Z][a-z]+(?:\s[a-z]+){0,2}|[a-z]{4,}(?:_[a-z]+)*)\b"
    )
    seen: Set[str] = set()
    entities: List[str] = []
    for m in pattern.finditer(text):
        c = _clean_concept(m.group(1))
        if c and c not in seen:
            seen.add(c)
            entities.append(c)
    return entities[:40]  # cap


def _confidence_from_verb(verb: str, context_len: int) -> float:
    """Base confidence, slightly reduced for short/ambiguous contexts."""
    base = 0.35
    # Directional verbs are more precise → slightly higher confidence
    if verb.lower() in ("increases", "decreases", "reduces", "improves"):
        base = 0.38
    # Very short sentences are less reliable
    if context_len < 30:
        base -= 0.04
    return round(min(PERCEPTION_CONFIDENCE_MAX, max(0.18, base)), 3)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ScreenTracker:
    """Asynchronous screen perception module.

    Parameters
    ----------
    knowledge_graph
        Live KG reference for node alignment.
    capture_interval
        Seconds between automatic captures.  Set to 0 to disable automatic
        capture (manual ``ingest_text()`` only).
    perception_log_path
        If given, each PerceptionEvent is appended to this JSONL file so the
        dashboard can read recent observations without touching the brain state.
    """

    def __init__(
        self,
        knowledge_graph=None,
        capture_interval: float = CAPTURE_INTERVAL_S,
        perception_log_path: Optional[Path] = None,
    ) -> None:
        self._graph              = knowledge_graph
        self._capture_interval   = capture_interval
        self._log_path           = perception_log_path

        # Thread-safe queues
        self._pending:           deque = deque()              # PerceptionTriple objects ready for KG
        self._episodic:          deque = deque(maxlen=EPISODIC_LOG_MAXLEN)  # recent PerceptionEvents
        self._lock               = threading.Lock()

        # Dedup: (subject, relation, object) → last_emitted_timestamp
        self._emitted:           Dict[Tuple[str, str, str], float] = {}

        # Metrics
        self.total_frames:       int = 0
        self.total_triples:      int = 0
        self.total_perception_edges: int = 0

        # Background thread
        self._running            = False
        self._paused             = False   # soft pause — thread keeps running but skips captures
        self._file_paused        = False   # last state read from the shared state file
        self._state_check_ts     = 0.0     # timestamp of last state-file poll
        self._thread: Optional[threading.Thread] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running:
            return
        if self._capture_interval <= 0:
            logger.info("[perception] auto-capture disabled (interval=0)")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="ScreenTracker"
        )
        self._thread.start()
        logger.info(
            "[perception] started interval=%.1fs", self._capture_interval
        )

    def stop(self) -> None:
        """Stop the background capture thread."""
        self._running = False

    def pause(self) -> None:
        """Pause automatic screen capture without killing the thread.

        The background thread keeps running but skips OCR.
        Call resume() to re-enable.  Safe to call multiple times.
        """
        self._paused = True
        logger.info("[perception] screen capture PAUSED")

    def resume(self) -> None:
        """Resume automatic screen capture after a pause."""
        self._paused = False
        logger.info("[perception] screen capture RESUMED")

    def is_paused(self) -> bool:
        """Return True if screen capture is currently paused (either source)."""
        return self._is_capture_paused()

    def ingest_text(self, text: str, source: str = "text_feed", tick: int = 0) -> List[PerceptionTriple]:
        """Process arbitrary text as a perception event.

        This is the primary entry point when no real display is available,
        or when the caller wants to feed text from a file, clipboard, or
        web page directly.

        Returns the list of extracted triples.
        """
        if not text or not text.strip():
            return []
        return self._process_text(text.strip(), source=source, tick=tick)

    def drain_pending(self) -> List[PerceptionTriple]:
        """Return and clear all triples queued for KG ingestion.

        Call this from the event loop tick body.
        """
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()
        return batch

    def get_recent_events(self, n: int = 20) -> List[PerceptionEvent]:
        """Return up to n most recent perception episodes."""
        with self._lock:
            return list(self._episodic)[-n:]

    def summary(self) -> Dict:
        """Metrics dict for the dashboard."""
        return {
            "frames":   self.total_frames,
            "triples":  self.total_triples,
            "kg_edges": self.total_perception_edges,
            "pending":  len(self._pending),
            "episodic": len(self._episodic),
            "paused":   self._paused,
        }

    # ── Capture ────────────────────────────────────────────────────────────

    def capture_screen(self) -> Optional[str]:
        """Grab the screen and return extracted text, or None if unavailable."""
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            return self.extract_text(img)
        except Exception as exc:
            logger.debug("[perception] capture_failed: %s", exc)
            return None

    def extract_text(self, image) -> Optional[str]:
        """Run OCR on a PIL Image and return the text."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image, lang="eng")
            return text.strip() if text else None
        except Exception as exc:
            logger.debug("[perception] ocr_failed: %s", exc)
            return None

    # ── Extraction ─────────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> List[str]:
        """Extract candidate concept names from text."""
        return _extract_entities_from_text(text)

    def extract_relations(self, text: str) -> List[PerceptionTriple]:
        """Extract structured causal triples from text.

        Only sentences containing recognised causal verbs are processed.
        Raw text is never injected; only normalised concept names are used.
        """
        triples: List[PerceptionTriple] = []
        seen_keys: Set[Tuple[str, str, str]] = set()

        for match in _RELATION_RE.finditer(text):
            subj_raw  = match.group(1)
            verb_raw  = match.group(2).lower()
            obj_raw   = match.group(3)

            subj = _clean_concept(subj_raw)
            obj  = _clean_concept(obj_raw)

            if not subj or not obj or subj == obj:
                continue
            if len(subj) < MIN_ENTITY_LENGTH or len(obj) < MIN_ENTITY_LENGTH:
                continue
            if verb_raw not in _CAUSAL_VERBS:
                continue

            relation, direction = _CAUSAL_VERBS[verb_raw]
            key = (subj, relation, obj)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Context window for confidence estimation
            start = max(0, match.start() - 20)
            end   = min(len(text), match.end() + 20)
            snippet = text[start:end].replace("\n", " ").strip()

            confidence = _confidence_from_verb(verb_raw, len(snippet))

            triples.append(PerceptionTriple(
                subject=subj,
                relation=relation,
                object=obj,
                confidence=confidence,
                direction=direction,
                source_text=snippet,
            ))

            if len(triples) >= MAX_TRIPLES_PER_FRAME:
                break

        return triples

    def align_nodes(self, triple: PerceptionTriple) -> PerceptionTriple:
        """Align triple entities to existing KG nodes when possible.

        If an existing node closely matches the extracted entity, reuse
        the existing node name to prevent synonym proliferation.
        """
        if self._graph is None:
            return triple

        def _best_match(concept: str) -> str:
            if concept in self._graph.nodes:
                return concept
            # Substring match: if concept is a prefix or suffix of a KG node
            for node in self._graph.nodes:
                if (len(concept) >= 4 and
                        (node.startswith(concept) or concept.startswith(node)
                         or node.endswith(concept) or concept.endswith(node))):
                    return node
            return concept

        aligned_subj = _best_match(triple.subject)
        aligned_obj  = _best_match(triple.object)

        if aligned_subj != triple.subject or aligned_obj != triple.object:
            logger.debug(
                "[perception] aligned %s→%s  %s→%s",
                triple.subject, aligned_subj, triple.object, aligned_obj,
            )
            triple = PerceptionTriple(
                subject=aligned_subj,
                relation=triple.relation,
                object=aligned_obj,
                confidence=triple.confidence,
                direction=triple.direction,
                source_text=triple.source_text,
                timestamp=triple.timestamp,
                tick=triple.tick,
            )
        return triple

    def ingest_perception(
        self,
        triple: PerceptionTriple,
        tick: int = 0,
    ) -> bool:
        """Queue a validated triple for KG ingestion.

        Returns True if the triple was newly queued (not a duplicate).

        Rules:
        - Stores in episodic memory always.
        - Queues for KG ingestion only if above confidence threshold and
          not a recent duplicate.
        """
        key = (triple.subject, triple.relation, triple.object)
        now = time.time()

        with self._lock:
            last = self._emitted.get(key, 0.0)
            if now - last < DUPLICATE_WINDOW:
                return False   # duplicate within window
            self._emitted[key] = now
            triple.tick = tick
            self._pending.append(triple)

        self.total_triples += 1
        logger.info(
            "[perception] triple %s -[%s]-> %s conf=%.2f src=%r",
            triple.subject, triple.relation, triple.object,
            triple.confidence, triple.source_text[:60],
        )
        return True

    # ── Internal ───────────────────────────────────────────────────────────

    def _process_text(
        self, text: str, source: str = "screen", tick: int = 0
    ) -> List[PerceptionTriple]:
        """Full pipeline: text → entities → relations → align → queue."""
        if not text:
            return []

        entities = self.extract_entities(text)
        raw_triples = self.extract_relations(text)

        queued: List[PerceptionTriple] = []
        for triple in raw_triples:
            triple = self.align_nodes(triple)
            if self.ingest_perception(triple, tick=tick):
                queued.append(triple)

        # Build and store the episode
        frame_hash = str(hash(text[:200]))
        event = PerceptionEvent(
            timestamp=time.time(),
            source=source,
            raw_text=text[:500],   # truncate for storage
            entities=entities[:20],
            triples=queued,
            frame_hash=frame_hash,
        )
        with self._lock:
            self._episodic.append(event)

        if self._log_path is not None and queued:
            self._write_event_log(event)

        self.total_frames += 1
        if queued:
            logger.info(
                "[perception] frame processed source=%s entities=%d triples=%d",
                source, len(entities), len(queued),
            )
        return queued

    def _write_event_log(self, event: PerceptionEvent) -> None:
        """Append a perception event to the JSONL log for the dashboard."""
        try:
            record = {
                "ts":       event.timestamp,
                "source":   event.source,
                "text":     event.raw_text[:200],
                "entities": event.entities[:10],
                "triples":  [
                    {
                        "subject":    t.subject,
                        "relation":   t.relation,
                        "object":     t.object,
                        "confidence": t.confidence,
                        "direction":  t.direction,
                    }
                    for t in event.triples
                ],
            }
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug("[perception] log_write_failed: %s", exc)

    def _sync_state_from_file(self) -> None:
        """Poll the shared state file and update _file_paused.

        Called at most once every _STATE_POLL_INTERVAL_S seconds so the
        hot path (inside the capture loop) stays cheap.
        """
        now = time.time()
        if now - self._state_check_ts < _STATE_POLL_INTERVAL_S:
            return
        self._state_check_ts = now
        try:
            if _SCREEN_STATE_PATH.exists():
                raw = _SCREEN_STATE_PATH.read_text(encoding="utf-8").strip()
                if raw:
                    data = json.loads(raw)
                    new_file_paused = not data.get("screen_capture_enabled", True)
                    if new_file_paused != self._file_paused:
                        self._file_paused = new_file_paused
                        logger.info(
                            "[perception] state_file -> capture %s",
                            "PAUSED" if new_file_paused else "RESUMED",
                        )
        except Exception as exc:
            logger.debug("[perception] state_file_read_error: %s", exc)

    def _is_capture_paused(self) -> bool:
        """Return True if capture should be skipped (either source)."""
        return self._paused or self._file_paused

    def _capture_loop(self) -> None:
        """Background thread: capture screen periodically."""
        logger.info("[perception] capture_loop started")
        while self._running:
            try:
                self._sync_state_from_file()   # cheap poll, no-ops if too soon
                if not self._is_capture_paused():
                    text = self.capture_screen()
                    if text and len(text.strip()) > 20:
                        self._process_text(text, source="screen")
            except Exception as exc:
                logger.debug("[perception] capture_loop_error: %s", exc)
            time.sleep(self._capture_interval)
        logger.info("[perception] capture_loop stopped")