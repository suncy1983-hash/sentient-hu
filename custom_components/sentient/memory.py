"""Long-term memory for Sentient.

Memories are short, durable facts ("the rabbits live in the hutch by the shed",
"Conah likes the kitchen light at 60% in the evening"). They are stored in Home
Assistant's own storage, shared by every voice and automation, and surfaced to
any conversation agent that uses the Sentient LLM API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import TYPE_CHECKING, Any
import unicodedata
import uuid

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from .semantic import SemanticIndex

# Semantic recall (multilingual-e5 cosines run high: unrelated ~0.75, related ~0.85+).
SEMANTIC_MIN = 0.80  # nothing below this counts as related
SEMANTIC_SPREAD = 0.06  # and nothing much further from the best match than this
KEYWORD_BONUS = 0.02  # a shared word nudges an equally close memory up

from .const import (  # noqa: E402
    DOMAIN,
    EVENT_MEMORY_CHANGED,
    MEMORY_CATEGORIES,
    MEMORY_MAX_ITEMS,
    MEMORY_STORAGE_VERSION,
)

# Any letters or digits, in any alphabet ("kút", "nyúlketrec"), with an optional apostrophe part.
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def _fold(text: str) -> str:
    """Lower case without accents, so 'Kút', 'kut' and 'kúthoz' all meet at 'kut'."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


_STOP = {_fold(w) for w in (
    # English
    "the", "a", "an", "is", "are", "was", "to", "of", "and", "in", "on", "at",
    "it", "my", "me", "i", "you", "do", "does", "what", "who", "when", "where",
    "for", "with", "that", "this", "about", "like", "likes",
    # Hungarian
    "az", "egy", "és", "is", "van", "vannak", "volt", "voltak", "lesz", "hogy", "nem", "de", "meg",
    "mi", "mit", "mik", "ki", "kit", "hol", "hova", "honnan", "mikor", "miért", "milyen", "melyik",
    "ez", "ezt", "azt", "ezek", "azok", "itt", "ott", "én", "te", "ő", "ti", "ők", "nekem", "neked",
    "nekünk", "nektek", "nála", "nálunk", "mint", "csak", "már", "még", "kell", "szeret", "szereti",
    "szeretjük", "pedig", "vagy", "akkor", "amikor", "mert", "ha", "sem", "se", "nagyon", "minden",
    "emlékezz", "emlékszel", "jegyezd", "tudod", "tudni", "rá", "erre", "arra", "róla",
)}


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(_fold(text)) if w not in _STOP and len(w) > 1}


def _related(a: str, b: str) -> bool:
    """Word forms of the same thing: rabbit/rabbits, nyul/nyulak/nyulketrec, kutya/kutyat."""
    if a == b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 3:
        return False
    if long.startswith(short):  # a stem with an ending: kut / kuthoz, rabbit / rabbits
        return True
    if len(short) >= 4 and short in long:  # part of a compound: ketrec / nyulketrec
        return True
    # a shared stem that is most of the shorter word: nyulak / nyulketrec, kavet / kaveja
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= 4 and common >= 0.6 * len(short)


@dataclass
class Memory:
    """A single remembered item."""

    text: str
    category: str = "fact"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: str = field(default_factory=lambda: dt_util.now().isoformat())
    source: str | None = None
    hits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """Persisted, de-duplicated memory list."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self.hass = hass
        self._store: Store[dict[str, Any]] = Store(
            hass, MEMORY_STORAGE_VERSION, f"{DOMAIN}.{entry_id}.memory"
        )
        self._items: list[Memory] = []
        self.semantic: SemanticIndex | None = None

    async def async_load(self) -> None:
        data = await self._store.async_load() or {}
        self._items = [Memory(**item) for item in data.get("items", [])]

    @callback
    def _save(self) -> None:
        self._store.async_delay_save(
            lambda: {"items": [m.as_dict() for m in self._items]}, 5
        )
        self.hass.bus.async_fire(EVENT_MEMORY_CHANGED, {"count": len(self._items)})

    @property
    def items(self) -> list[Memory]:
        return list(self._items)

    def add(self, text: str, category: str = "fact", source: str | None = None) -> Memory:
        """Add a memory, replacing a near-duplicate rather than piling up copies."""
        text = " ".join(text.split()).strip()
        if not text:
            raise ValueError("Nothing to remember")
        if category not in MEMORY_CATEGORIES:
            category = "fact"
        new_words = _words(text)
        for existing in self._items:
            old_words = _words(existing.text)
            if new_words and old_words:
                overlap = len(new_words & old_words) / len(new_words | old_words)
                if overlap >= 0.8:
                    existing.text = text
                    existing.category = category
                    existing.created = dt_util.now().isoformat()
                    self._save()
                    return existing
        memory = Memory(text=text, category=category, source=source)
        self._items.append(memory)
        if len(self._items) > MEMORY_MAX_ITEMS:
            # Drop the least useful old notes first.
            self._items.sort(key=lambda m: (m.category != "note", m.hits, m.created))
            self._items = self._items[-MEMORY_MAX_ITEMS:]
        self._save()
        return memory

    def get(self, memory_id: str) -> Memory | None:
        return next((m for m in self._items if m.id == memory_id), None)

    def update(self, memory_id: str, text: str, category: str | None = None) -> Memory | None:
        """Edit a memory in place (keeps its id and usage count)."""
        memory = self.get(memory_id)
        text = " ".join(text.split()).strip()
        if memory is None or not text:
            return None
        memory.text = text
        if category in MEMORY_CATEGORIES:
            memory.category = category
        self._save()
        return memory

    def forget(self, memory_id_or_text: str) -> Memory | None:
        """Remove by id, or by the best text match."""
        target = next((m for m in self._items if m.id == memory_id_or_text), None)
        if target is None:
            found = self.search(memory_id_or_text, limit=1, count_hits=False)
            target = found[0] if found else None
        if target is not None:
            self._items.remove(target)
            self._save()
        return target

    def _keyword_scores(self, query: str) -> dict[str, float]:
        q = _words(query)
        scores: dict[str, float] = {}
        for memory in self._items:
            words = _words(memory.text)
            score = len(q & words)
            # partial matches (rabbit / rabbits, nyúl / nyulakat / nyúlketrec)
            score += 0.5 * sum(1 for a in q for b in words if _related(a, b))
            if score:
                scores[memory.id] = score
        return scores

    def _count(self, result: list[Memory], count_hits: bool) -> list[Memory]:
        if count_hits and result:
            for m in result:
                m.hits += 1
            self._save()
        return result

    def search(self, query: str, limit: int = 8, count_hits: bool = True) -> list[Memory]:
        """Keyword-ranked recall. Good enough for hundreds of short facts."""
        if not _words(query):
            return self.recent(limit)
        scores = self._keyword_scores(query)
        ranked = sorted((m for m in self._items if m.id in scores), key=lambda m: (scores[m.id], m.created), reverse=True)
        return self._count(ranked[:limit], count_hits)

    async def async_search(self, query: str, limit: int = 8, count_hits: bool = True) -> list[Memory]:
        """Recall by meaning and words together; keywords alone if the text model isn't available.

        "Mit eszik a kutya?" finds "Bodri csak a száraz tápot szereti" though no word is shared.
        """
        if not query.strip():
            return self.recent(limit)
        meaning = await self.semantic.async_rank(query) if self.semantic is not None else None
        if not meaning:
            return self.search(query, limit, count_hits)
        words = self._keyword_scores(query)
        best = max(meaning.values())
        floor = max(SEMANTIC_MIN, best - SEMANTIC_SPREAD)
        scored: list[tuple[float, Memory]] = []
        for m in self._items:
            cos, kw = meaning.get(m.id), words.get(m.id, 0.0)
            if (cos is not None and cos >= floor) or kw:
                scored.append(((cos if cos is not None else floor) + KEYWORD_BONUS * min(kw, 3), m))
        scored.sort(key=lambda s: (s[0], s[1].created), reverse=True)
        return self._count([m for _, m in scored[:limit]], count_hits)

    async def async_forget(self, memory_id_or_text: str) -> tuple[Memory | None, list[Memory]]:
        """Forget by id, or by a match close enough to be sure; otherwise return the closest few to ask about."""
        target = self.get(memory_id_or_text)
        if target is None:
            found = await self.async_search(memory_id_or_text, limit=3, count_hits=False)
            kw = self._keyword_scores(memory_id_or_text)
            top = kw.get(found[0].id, 0) if found else 0
            second = kw.get(found[1].id, 0) if len(found) > 1 else 0
            sure = bool(found) and (len(found) == 1 or (top >= 2 and top > second))
            if not sure:
                return None, found
            target = found[0]
        self._items.remove(target)
        self._save()
        return target, []

    def recent(self, limit: int) -> list[Memory]:
        return sorted(self._items, key=lambda m: (m.hits, m.created), reverse=True)[:limit]
