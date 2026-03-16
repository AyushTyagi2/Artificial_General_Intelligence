"""WikiReader — clean Wikipedia perception channel for digital_baby.

Fetches Wikipedia article text directly via the REST API and extracts causal
relations into structured triples, injecting them into the knowledge graph as a
second perception channel alongside ScreenTracker.

Pipeline
--------
ingest_topic(topic)
  → fetch_article()      (Wikipedia REST API, plain-text endpoint)
  → clean_text()         (strip markup noise)
  → split_sentences()    (regex; later spaCy)
  → extract_relations()  (same causal-verb table as ScreenTracker)
  → queue triples        (drain_pending() feeds the event loop)

Design constraints
------------------
* Never injects raw text into the knowledge graph.
* Confidence band 0.45–0.55 — higher than OCR, below experimental confirmation.
* Runs asynchronously; does not block the event loop.
* Hard safety caps: MAX_SENTENCES_PER_ARTICLE, MAX_TRIPLES_PER_ARTICLE.
* Rate-limited: no more than one article fetch per MIN_FETCH_INTERVAL_TICKS ticks.

Integration
-----------
In BabyEventLoop.__init__:
    from digital_baby.brain.perception import WikiReader
    self.wiki_reader = WikiReader(knowledge_graph=self.knowledge_graph)
    self.wiki_reader.start()

Every ~50 ticks (topic selection):
    topic = self._pick_wiki_topic()
    self.wiki_reader.ingest_topic(topic)

Every tick (drain):
    wiki_triples = self.wiki_reader.drain_pending()
    for triple in wiki_triples:
        self.knowledge_graph.add_or_update_edge(
            triple.subject, triple.object, triple.relation,
            confidence=triple.confidence,
            provenance="wikipedia",
            current_tick=tick,
        )
        self.ingestion.suggest_concepts([triple.subject, triple.object])
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIKI_API_BASE              = "https://en.wikipedia.org/api/rest_v1/page/plain/{title}"
WIKI_CONFIDENCE_MIN: float = 0.45
WIKI_CONFIDENCE_MAX: float = 0.55
MIN_ENTITY_LENGTH:   int   = 3
MAX_ARTICLE_CHARS:   int   = 50_000    # truncate to prevent memory spikes
MAX_SENTENCES_PER_ARTICLE: int = 400
MAX_TRIPLES_PER_ARTICLE:   int = 50
MIN_SENTENCE_LEN:    int   = 40
MAX_SENTENCE_LEN:    int   = 400
MIN_FETCH_INTERVAL_TICKS: int = 30     # rate-limit: ticks between article fetches
DUPLICATE_WINDOW_S:  float = 120.0     # seconds before re-emitting same triple
EPISODIC_LOG_MAXLEN: int   = 100

# Identical causal-verb mapping as ScreenTracker for consistency
_CAUSAL_VERBS: Dict[str, Tuple[str, str]] = {
    "increases":   ("positive_affects", "positive"),
    "improves":    ("positive_affects", "positive"),
    "promotes":    ("positive_affects", "positive"),
    "enhances":    ("positive_affects", "positive"),
    "activates":   ("positive_affects", "positive"),
    "stimulates":  ("positive_affects", "positive"),
    "boosts":      ("positive_affects", "positive"),
    "accelerates": ("positive_affects", "positive"),
    "decreases":   ("negative_affects", "negative"),
    "reduces":     ("negative_affects", "negative"),
    "inhibits":    ("negative_affects", "negative"),
    "suppresses":  ("negative_affects", "negative"),
    "slows":       ("negative_affects", "negative"),
    "impairs":     ("negative_affects", "negative"),
    "limits":      ("negative_affects", "negative"),
    "blocks":      ("negative_affects", "negative"),
    "causes":      ("affects",          "mixed"),
    "affects":     ("affects",          "mixed"),
    "influences":  ("affects",          "mixed"),
    "controls":    ("affects",          "mixed"),
    "regulates":   ("affects",          "mixed"),
    "modulates":   ("affects",          "mixed"),
    "drives":      ("affects",          "mixed"),
    "triggers":    ("affects",          "mixed"),
}

_VERB_ALT = "|".join(sorted(_CAUSAL_VERBS.keys(), key=len, reverse=True))
_RELATION_RE = re.compile(
    r"(?:^|(?<=\.\s)|(?<=\n)|\b)"
    r"((?:[A-Za-z][a-z_]*(?:[\s_][a-z]+){0,3}))"       # subject (1-4 words)
    r"\s+"
    r"(?:can\s+|may\s+|will\s+|also\s+|directly\s+|indirectly\s+)?"
    r"(" + _VERB_ALT + r")"                              # verb
    r"\s+"
    r"(?:the\s+|a\s+|an\s+|its\s+)?"
    r"((?:[A-Za-z][a-z_]*(?:[\s_][a-z]+){0,3}))",       # object (1-4 words)
    re.IGNORECASE,
)

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
class WikiTriple:
    """One structured causal relation extracted from a Wikipedia sentence."""
    subject:     str
    relation:    str
    object:      str
    confidence:  float
    direction:   str         # "positive" | "negative" | "mixed"
    source_text: str         # the sentence this triple was derived from
    timestamp:   float = field(default_factory=time.time)
    tick:        int   = 0


@dataclass
class WikiArticleEvent:
    """One Wikipedia article ingest episode stored for dashboard display."""
    timestamp:   float
    topic:       str
    sentences:   int
    triples:     int
    sample_triples: List[WikiTriple]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def clean_entity(text: str) -> Optional[str]:
    """Normalise a raw phrase to a canonical concept name.

    Rules:
    - lowercase
    - replace spaces/hyphens with underscores
    - strip non-alphanumeric chars
    - reject if shorter than MIN_ENTITY_LENGTH
    - reject stop words
    - reject pure digit strings
    """
    c = text.strip().lower().replace(" ", "_").replace("-", "_")
    c = re.sub(r"[^a-z0-9_]", "", c)
    c = re.sub(r"_+", "_", c).strip("_")
    if len(c) < MIN_ENTITY_LENGTH:
        return None
    # Drop trailing 's' noise for very short words that slip through
    base = c.rstrip("s") if len(c) > 4 else c
    if base in _STOP_WORDS or c in _STOP_WORDS:
        return None
    if c.replace("_", "").isdigit():
        return None
    return c


def _confidence_from_verb(verb: str, sentence_len: int) -> float:
    """Derive confidence score from verb precision and sentence quality."""
    if verb.lower() in ("increases", "decreases", "reduces", "stimulates",
                         "inhibits", "promotes", "suppresses"):
        base = WIKI_CONFIDENCE_MAX          # directional verbs are precise
    elif verb.lower() in ("causes", "triggers", "regulates"):
        base = (WIKI_CONFIDENCE_MIN + WIKI_CONFIDENCE_MAX) / 2
    else:
        base = WIKI_CONFIDENCE_MIN

    # Short sentences are less reliable
    if sentence_len < 50:
        base -= 0.03

    return round(min(WIKI_CONFIDENCE_MAX, max(WIKI_CONFIDENCE_MIN, base)), 3)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WikiReader:
    """Asynchronous Wikipedia perception module.

    Parameters
    ----------
    knowledge_graph
        Live KG reference for node alignment.
    wiki_log_path
        If given, each WikiArticleEvent is appended to this JSONL file so the
        dashboard can display Wikipedia-sourced relations separately.
    """

    def __init__(
        self,
        knowledge_graph=None,
        wiki_log_path=None,
    ) -> None:
        self._graph        = knowledge_graph
        self._log_path     = wiki_log_path

        # Thread-safe state
        self._pending:     deque = deque()
        self._episodic:    deque = deque(maxlen=EPISODIC_LOG_MAXLEN)
        self._lock         = threading.Lock()
        self._fetch_lock   = threading.Lock()

        # Dedup: (subject, relation, object) → last_emitted_timestamp
        self._emitted: Dict[Tuple[str, str, str], float] = {}

        # Rate-limiting
        self._last_fetch_tick: int = -MIN_FETCH_INTERVAL_TICKS

        # Metrics
        self.total_articles:     int = 0
        self.total_triples:      int = 0
        self.total_wiki_edges:   int = 0

        # Background thread (ingest queue)
        self._running = False
        self._ingest_queue: deque = deque()
        self._thread: Optional[threading.Thread] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background ingest worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="WikiReader"
        )
        self._thread.start()
        logger.info("[wiki] reader started")

    def stop(self) -> None:
        """Stop the background worker thread."""
        self._running = False

    def ingest_topic(self, topic: str, current_tick: int = 0) -> None:
        """Queue a Wikipedia topic for background ingestion.

        Respects the rate-limit: no-ops if called more frequently than
        MIN_FETCH_INTERVAL_TICKS ticks since last fetch.
        """
        if current_tick - self._last_fetch_tick < MIN_FETCH_INTERVAL_TICKS:
            logger.debug(
                "[wiki] rate_limit: skipping topic=%r (last_fetch_tick=%d, current=%d)",
                topic, self._last_fetch_tick, current_tick,
            )
            return
        self._last_fetch_tick = current_tick
        with self._fetch_lock:
            self._ingest_queue.append((topic, current_tick))
        logger.debug("[wiki] queued topic=%r tick=%d", topic, current_tick)

    def drain_pending(self) -> List[WikiTriple]:
        """Return and clear all triples queued for KG ingestion.

        Call this from the event loop tick body every tick.
        """
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()
        return batch

    def get_recent_events(self, n: int = 20) -> List[WikiArticleEvent]:
        """Return up to n most recent article ingest episodes."""
        with self._lock:
            return list(self._episodic)[-n:]

    def summary(self) -> Dict:
        """Metrics dict for the dashboard."""
        return {
            "articles":  self.total_articles,
            "triples":   self.total_triples,
            "kg_edges":  self.total_wiki_edges,
            "pending":   len(self._pending),
            "episodic":  len(self._episodic),
        }

    # ── Core pipeline ───────────────────────────────────────────────────────

    def fetch_article(self, title: str) -> Optional[str]:
        """Fetch plain-text content of a Wikipedia article.

        Uses the Wikipedia REST v1 plain-text endpoint.
        Returns None on any error.
        """
        url = WIKI_API_BASE.format(title=urllib.parse.quote(title, safe=""))
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "DigitalBrainAgent/1.0 (educational AI research)",
                    "Accept": "text/plain",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.debug("[wiki] fetch_failed topic=%r status=%d", title, resp.status)
                    return None
                raw = resp.read(MAX_ARTICLE_CHARS + 1024)
                text = raw.decode("utf-8", errors="replace")
                if not text or not text.strip():
                    logger.debug("[wiki] empty_article topic=%r", title)
                    return None
                return text[:MAX_ARTICLE_CHARS]
        except urllib.error.HTTPError as exc:
            logger.debug("[wiki] http_error topic=%r status=%d", title, exc.code)
            return None
        except Exception as exc:
            logger.debug("[wiki] fetch_error topic=%r: %s", title, exc)
            return None

    def clean_text(self, text: str) -> str:
        """Strip Wikipedia markup noise from plain text."""
        # Remove section headers (== Header ==)
        text = re.sub(r"={2,}[^=]+=+", " ", text)
        # Remove wiki-style references [1], [note 3], etc.
        text = re.sub(r"\[\d+\]|\[note\s*\d+\]", "", text)
        # Remove parenthetical pronunciations / IPA
        text = re.sub(r"\(\/[^)]+\/\)", "", text)
        # Collapse excessive whitespace
        text = re.sub(r"\s{2,}", " ", text)
        # Remove lines that look like pure metadata / tables
        lines = [l for l in text.splitlines() if len(l.strip()) > 30]
        return " ".join(lines)

    def split_sentences(self, text: str) -> List[str]:
        """Split article text into filtered sentences."""
        raw = re.split(r"[.!?]", text)
        sentences = []
        for s in raw:
            s = s.strip()
            if MIN_SENTENCE_LEN <= len(s) <= MAX_SENTENCE_LEN:
                sentences.append(s)
            if len(sentences) >= MAX_SENTENCES_PER_ARTICLE:
                break
        return sentences

    def extract_relations(self, text: str) -> List[WikiTriple]:
        """Extract causal triples from article text using the shared verb table."""
        triples: List[WikiTriple] = []
        seen_keys: Set[Tuple[str, str, str]] = set()

        for match in _RELATION_RE.finditer(text):
            subj_raw = match.group(1)
            verb_raw = match.group(2).lower()
            obj_raw  = match.group(3)

            subj = clean_entity(subj_raw)
            obj  = clean_entity(obj_raw)

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

            start   = max(0, match.start() - 15)
            end     = min(len(text), match.end() + 15)
            snippet = text[start:end].replace("\n", " ").strip()

            confidence = _confidence_from_verb(verb_raw, len(snippet))

            triples.append(WikiTriple(
                subject=subj,
                relation=relation,
                object=obj,
                confidence=confidence,
                direction=direction,
                source_text=snippet,
            ))

            if len(triples) >= MAX_TRIPLES_PER_ARTICLE:
                break

        return triples

    def align_nodes(self, triple: WikiTriple) -> WikiTriple:
        """Align triple entities to existing KG nodes to prevent synonym sprawl."""
        if self._graph is None:
            return triple

        def _best_match(concept: str) -> str:
            if concept in self._graph.nodes:
                return concept
            for node in self._graph.nodes:
                if (len(concept) >= 4 and (
                        node.startswith(concept) or concept.startswith(node)
                        or node.endswith(concept) or concept.endswith(node))):
                    return node
            return concept

        aligned_subj = _best_match(triple.subject)
        aligned_obj  = _best_match(triple.object)

        if aligned_subj != triple.subject or aligned_obj != triple.object:
            logger.debug(
                "[wiki] aligned %s→%s  %s→%s",
                triple.subject, aligned_subj, triple.object, aligned_obj,
            )
            triple = WikiTriple(
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

    def _queue_triple(self, triple: WikiTriple, tick: int = 0) -> bool:
        """Queue a validated triple for KG ingestion.

        Returns True if newly queued (not a recent duplicate).
        """
        key = (triple.subject, triple.relation, triple.object)
        now = time.time()

        with self._lock:
            last = self._emitted.get(key, 0.0)
            if now - last < DUPLICATE_WINDOW_S:
                return False
            self._emitted[key] = now
            triple.tick = tick
            self._pending.append(triple)

        self.total_triples += 1
        logger.info(
            "[wiki] triple %s -[%s]-> %s conf=%.2f",
            triple.subject, triple.relation, triple.object, triple.confidence,
        )
        return True

    def _process_topic(self, topic: str, tick: int = 0) -> Optional[WikiArticleEvent]:
        """Full pipeline: fetch → clean → sentences → relations → queue."""
        logger.info("[wiki] fetching article=%r", topic)
        raw_text = self.fetch_article(topic)
        if not raw_text:
            logger.debug("[wiki] no_text topic=%r", topic)
            return None

        cleaned   = self.clean_text(raw_text)
        sentences = self.split_sentences(cleaned)

        # Extract relations across the whole cleaned text (not sentence-by-sentence)
        # so the regex can match multi-word subjects that span sentence boundaries.
        all_triples_raw = self.extract_relations(cleaned)

        queued_triples: List[WikiTriple] = []
        for triple in all_triples_raw:
            triple = self.align_nodes(triple)
            if self._queue_triple(triple, tick=tick):
                queued_triples.append(triple)

        self.total_articles += 1
        logger.info(
            "[wiki] article=%s sentences=%d triples=%d",
            topic, len(sentences), len(queued_triples),
        )

        event = WikiArticleEvent(
            timestamp=time.time(),
            topic=topic,
            sentences=len(sentences),
            triples=len(queued_triples),
            sample_triples=queued_triples[:5],
        )

        with self._lock:
            self._episodic.append(event)

        if self._log_path is not None and queued_triples:
            self._write_event_log(event)

        return event

    # ── Logging ─────────────────────────────────────────────────────────────

    def _write_event_log(self, event: WikiArticleEvent) -> None:
        """Append a wiki ingest event to the JSONL log for the dashboard."""
        try:
            record = {
                "ts":       event.timestamp,
                "source":   "wikipedia",
                "topic":    event.topic,
                "sentences": event.sentences,
                "triples_count": event.triples,
                "triples":  [
                    {
                        "subject":    t.subject,
                        "relation":   t.relation,
                        "object":     t.object,
                        "confidence": t.confidence,
                        "direction":  t.direction,
                        "text":       t.source_text[:100],
                    }
                    for t in event.sample_triples
                ],
            }
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.debug("[wiki] log_write_failed: %s", exc)

    # ── Background worker ───────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Background thread: drain the ingest queue one topic at a time."""
        logger.info("[wiki] worker_loop started")
        while self._running:
            try:
                with self._fetch_lock:
                    item = self._ingest_queue.popleft() if self._ingest_queue else None
                if item is not None:
                    topic, tick = item
                    self._process_topic(topic, tick=tick)
            except IndexError:
                pass
            except Exception as exc:
                logger.debug("[wiki] worker_loop_error: %s", exc)
            time.sleep(0.5)
        logger.info("[wiki] worker_loop stopped")


# Import needed for URL encoding in fetch_article
import urllib.parse  # noqa: E402 (placed after class to avoid circular at top)