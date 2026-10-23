"""Who is speaking — Sentient's side of the Sentient Voice ID app.

The voice STT (stt.py) hands every utterance to ``async_identify`` while the real speech-to-text works on
it, so by the time the conversation agent runs, ``current()`` says who said it. Voices are taught by
voice: "learn my voice, I'm Sanya" enrols the sentence just heard plus the next few; "learn Lili's
voice" enrols only the next few (so the parent asking isn't taken for Lili).
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import logging
from typing import Any

import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from .const import CONF_VOICE_ID_URL, EVENT_SPEAKER

_LOGGER = logging.getLogger(__name__)

FRESH = timedelta(seconds=45)  # who spoke last counts for this long
ENROL_WINDOW = timedelta(minutes=3)
ENROL_UTTERANCES = 5
MIN_BYTES = int(16000 * 2 * 0.6)  # 0.6 s of 16 kHz 16-bit audio
_TIMEOUT = aiohttp.ClientTimeout(total=4)


class VoiceIDError(Exception):
    """The Voice ID app said no, or couldn't be reached."""


@dataclass
class Heard:
    name: str | None
    known: bool
    unsure: bool
    score: float
    at: datetime
    print_id: str | None = None
    enrolling: bool = False

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["at"] = self.at.isoformat()
        return out


class VoiceID:
    def __init__(self, hass: HomeAssistant, options: dict[str, Any]) -> None:
        self.hass = hass
        self.options = options
        self.last: Heard | None = None
        self._enrol: dict[str, Any] | None = None

    @property
    def url(self) -> str:
        return str(self.options.get(CONF_VOICE_ID_URL) or "").strip().rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self.enabled:
            raise VoiceIDError("voice recognition isn't set up (no Voice ID address in Sentient's settings)")
        session = async_get_clientsession(self.hass)
        try:
            async with session.request(method, f"{self.url}{path}", timeout=_TIMEOUT, **kwargs) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise VoiceIDError(text or f"HTTP {resp.status}")
                return json.loads(text) if text else {}
        except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as err:
            raise VoiceIDError(f"the Voice ID app can't be reached: {err}") from err

    # ----- every utterance -----------------------------------------------------------------
    def enrolling(self) -> dict[str, Any] | None:
        if self._enrol and (self._enrol["left"] <= 0 or dt_util.utcnow() > self._enrol["until"]):
            self._enrol = None
        return self._enrol

    async def async_identify(self, pcm: bytes) -> Heard | None:
        """Who said this (16 kHz mono 16-bit PCM)? Never raises: speech must go on without it."""
        if not self.enabled or len(pcm) < MIN_BYTES:
            return None
        enrol = self.enrolling()
        try:
            res = await self._call("POST", "/identify" + ("?learn=0" if enrol else ""), data=pcm,
                                   headers={"Content-Type": "application/octet-stream"})
        except VoiceIDError as err:
            _LOGGER.debug("Voice not identified: %s", err)
            return None
        heard = Heard(name=res.get("name"), known=bool(res.get("known")), unsure=bool(res.get("unsure")),
                      score=float(res.get("score") or 0), at=dt_util.utcnow(), print_id=res.get("id"))
        if enrol and heard.print_id:
            try:
                await self._call("POST", "/enroll", json={"name": enrol["name"], "ids": [heard.print_id]})
                enrol["left"] -= 1
                heard = Heard(name=enrol["name"], known=True, unsure=False, score=1.0, at=heard.at,
                              print_id=heard.print_id, enrolling=True)
            except VoiceIDError as err:
                _LOGGER.warning("Could not learn %s's voice: %s", enrol["name"], err)
        self.last = heard
        self.hass.bus.async_fire(EVENT_SPEAKER, heard.as_dict())
        return heard

    def current(self) -> Heard | None:
        """The last speaker, if it was just now (so it belongs to the conversation that's running)."""
        if self.last and dt_util.utcnow() - self.last.at <= FRESH:
            return self.last
        return None

    # ----- teaching and forgetting ------------------------------------------------------------
    async def async_learn(self, name: str, this_is_them: bool) -> dict[str, Any]:
        name = " ".join(str(name).split()).strip()
        if not name:
            raise VoiceIDError("whose voice? a name is needed")
        taken = 0
        if this_is_them and (last := self.current()) and last.print_id:
            await self._call("POST", "/enroll", json={"name": name, "ids": [last.print_id]})
            taken = 1
        self._enrol = {"name": name, "until": dt_util.utcnow() + ENROL_WINDOW, "left": ENROL_UTTERANCES - taken}
        return {"learning": name, "already_heard": taken, "sentences_still_needed": self._enrol["left"],
                "minutes": int(ENROL_WINDOW.total_seconds() // 60)}

    def stop_learning(self) -> None:
        self._enrol = None

    async def async_people(self) -> list[dict[str, Any]]:
        return await self._call("GET", "/people")

    async def async_forget(self, name: str) -> dict[str, Any]:
        return await self._call("DELETE", f"/people/{name}")
