"""Semantic memory: finding memories by meaning, not just by shared words.

Each memory gets a vector from the Sentient app's text model (multilingual-e5, local, Hungarian
included). Vectors live in their own store, keyed by memory id and a hash of the text, and are worked out
in the background whenever memories change — so adding a memory never waits for the model, and nothing
breaks if the app is away: search simply falls back to keywords until it is back.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from typing import TYPE_CHECKING, Any

import aiohttp

from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.storage import Store

from .const import CONF_EMBED_URL, CONF_VOICE_ID_URL, DOMAIN, EVENT_MEMORY_CHANGED

if TYPE_CHECKING:
    from .memory import Memory

_LOGGER = logging.getLogger(__name__)

BATCH = 32
SYNC_DELAY = 3  # seconds after a change, so a burst of edits is embedded once
_TIMEOUT = aiohttp.ClientTimeout(total=20)


class EmbedError(Exception):
    """The text model couldn't be reached or said no."""


def _hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()[:12]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Embedder:
    """Talks to the app's POST /embed."""

    def __init__(self, hass: HomeAssistant, options: dict[str, Any]) -> None:
        self.hass = hass
        self.options = options

    @property
    def url(self) -> str:
        url = self.options.get(CONF_EMBED_URL) or self.options.get(CONF_VOICE_ID_URL) or ""
        return str(url).strip().rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def async_embed(self, texts: list[str], kind: str) -> tuple[str, list[list[float]]]:
        if not self.enabled:
            raise EmbedError("semantic memory isn't set up")
        session = async_get_clientsession(self.hass)
        try:
            async with session.post(f"{self.url}/embed", json={"texts": texts, "kind": kind}, timeout=_TIMEOUT) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise EmbedError(body or f"HTTP {resp.status}")
                data = json.loads(body)
        except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError, ValueError) as err:
            raise EmbedError(f"the text model can't be reached: {err}") from err
        vectors = data.get("vectors") or []
        if len(vectors) != len(texts):
            raise EmbedError("the text model returned the wrong number of vectors")
        return str(data.get("model") or ""), vectors


class SemanticIndex:
    """Vectors for every memory, kept in step with the memory store."""

    def __init__(self, hass: HomeAssistant, entry_id: str, embedder: Embedder, items) -> None:
        self.hass = hass
        self.embedder = embedder
        self._items = items  # callable -> list[Memory]
        self._store: Store[dict[str, Any]] = Store(hass, 1, f"{DOMAIN}.{entry_id}.memory_vectors")
        self.model = ""
        self.vectors: dict[str, dict[str, Any]] = {}  # memory id -> {"h": text hash, "v": [floats]}
        self._timer: CALLBACK_TYPE | None = None
        self._unsub: CALLBACK_TYPE | None = None
        self._syncing = False
        self._reembed = False
        self.last_error: str | None = None

    async def async_load(self) -> None:
        data = await self._store.async_load() or {}
        self.model = data.get("model", "")
        self.vectors = data.get("items", {})

    @callback
    def async_start(self) -> None:
        self._unsub = self.hass.bus.async_listen(EVENT_MEMORY_CHANGED, self._changed)
        self.schedule()

    @callback
    def async_stop(self) -> None:
        if self._unsub:
            self._unsub()
            self._unsub = None
        if self._timer:
            self._timer()
            self._timer = None

    @callback
    def _changed(self, _event: Event) -> None:
        self.schedule()

    @callback
    def schedule(self, delay: float = SYNC_DELAY) -> None:
        if not self.embedder.enabled:
            return
        if self._timer:
            self._timer()

        @callback
        def _run(_now) -> None:
            self._timer = None
            self.hass.async_create_task(self.async_sync())

        self._timer = async_call_later(self.hass, delay, _run)

    def coverage(self) -> tuple[int, int]:
        items = self._items()
        done = sum(1 for m in items if (v := self.vectors.get(m.id)) and v.get("h") == _hash(m.text))
        return done, len(items)

    async def async_sync(self) -> int:
        """Embed every memory that is new or has changed; drop vectors of forgotten ones. Returns how many."""
        if self._syncing or not self.embedder.enabled:
            return 0
        self._syncing = True
        try:
            items: list[Memory] = self._items()
            ids = {m.id for m in items}
            before = len(self.vectors)
            self.vectors = {k: v for k, v in self.vectors.items() if k in ids}
            dropped = before != len(self.vectors)
            if self._reembed:
                self._reembed, self.model = False, ""
                self.vectors = {}
            todo = [m for m in items if (v := self.vectors.get(m.id)) is None or v.get("h") != _hash(m.text)]
            done = 0
            i = 0
            while i < len(todo):
                batch = todo[i:i + BATCH]
                model, vecs = await self.embedder.async_embed([m.text for m in batch], "passage")
                if model and self.model and model != self.model:
                    # a different model: old vectors don't compare with new ones, so start again
                    self.vectors, todo, i, done = {}, list(items), 0, 0
                    self.model = model
                    continue
                self.model = model or self.model
                for m, v in zip(batch, vecs):
                    self.vectors[m.id] = {"h": _hash(m.text), "v": v}
                done += len(batch)
                i += BATCH
            if done or dropped:
                self._store.async_delay_save(lambda: {"model": self.model, "items": self.vectors}, 2)
            self.last_error = None
            return done
        except EmbedError as err:
            self.last_error = str(err)
            _LOGGER.debug("Semantic memory not updated: %s", err)
            self.schedule(300)  # try again in five minutes
            return 0
        finally:
            self._syncing = False

    async def async_rank(self, query: str) -> dict[str, float] | None:
        """How close each memory's meaning is to the query (cosine), or None if that can't be worked out."""
        if not self.embedder.enabled or not self.vectors:
            return None
        try:
            model, vecs = await self.embedder.async_embed([query], "query")
        except EmbedError as err:
            _LOGGER.debug("Semantic search unavailable: %s", err)
            return None
        if model and self.model and model != self.model:
            self._reembed = True  # the app has a new model: work every memory out again
            self.schedule(0)
            return None
        qv = vecs[0]
        texts = {m.id: _hash(m.text) for m in self._items()}
        return {
            mid: cosine(qv, rec["v"]) for mid, rec in self.vectors.items()
            if texts.get(mid) == rec.get("h")
        }
