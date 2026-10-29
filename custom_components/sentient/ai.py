"""One door for the AI calls Sentient makes on its own.

* **Meter** — every background AI call is counted by what it was for (camera checks, briefings,
  the nightly think, the doorbell…), today and for the last fortnight. The Mind card shows it.
* **Daily cap** — once today's limit is used up, Sentient stops calling the AI until midnight and
  falls back to plain rules: briefings, welcomes and the doorbell still speak, in plain words;
  camera looks wait for tomorrow. A question someone asks still gets an answer (it's counted, never
  blocked). 0 means no limit.
* **Backup** — every call has a time limit, and a failure (an error, a timeout, an outage) is
  caught: the caller gets `AIUnavailable` and says its plain version instead. Three failures in a
  row show up as a house-health problem until the AI answers again.

Questions people ask a voice agent go through that agent, not through here, and never count.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    CONF_AI_TASK_ENTITY,
    CONF_BUDGET_CAMERAS,
    CONF_BUDGET_CLIPS,
    CONF_BUDGET_DRIVE,
    DEFAULT_BUDGET_CLIPS,
    CONF_BUDGET_NOTICING,
    CONF_DAILY_AI_LIMIT,
    DEFAULT_DAILY_AI_LIMIT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

TIMEOUT = 90  # seconds; picture questions can be slow
RETRY_AFTER = 2  # seconds before asking a local model a second time
FAILING_AFTER = 3  # failures in a row
HISTORY_DAYS = 14


def ai_task_entity(hass: HomeAssistant, options: dict[str, Any]) -> str | None:
    """Sentient's own choice; else Home Assistant's preferred one; else the first there is.

    Without this, a house that never picked a preferred AI Task gets "No entity_id provided and no
    preferred entity set" on every briefing.
    """
    if chosen := options.get(CONF_AI_TASK_ENTITY):
        return chosen
    try:
        from homeassistant.components.ai_task.const import DATA_PREFERENCES  # noqa: PLC0415

        prefs = hass.data.get(DATA_PREFERENCES)
        if prefs is not None and prefs.gen_data_entity_id:
            return None  # Home Assistant will use its own preferred one
    except ImportError:
        pass
    return next(iter(hass.states.async_entity_ids("ai_task")), None)


def _in_house_language(data: dict[str, Any]) -> dict[str, Any]:
    """Every background prompt asks for the house's language (some callers already do)."""
    from .i18n import write_in  # noqa: PLC0415

    text = data.get("instructions")
    if isinstance(text, str) and write_in().strip() not in text:
        data = {**data, "instructions": text + write_in()}
    return data


async def _forgiving_call(hass: HomeAssistant, data: dict[str, Any]) -> Any:
    """ai_task.generate_data, a little more forgiving than calling it straight.

    A local model sometimes answers with nothing at all (busy, or it only "thought"), which Home
    Assistant reports as an error. Try once more; if that fails too and the house has another AI Task,
    let that one answer this time.
    """

    data = _in_house_language(data)

    async def call(payload: dict[str, Any]) -> Any:
        return await hass.services.async_call("ai_task", "generate_data", payload, blocking=True, return_response=True)

    try:
        return await call(data)
    except HomeAssistantError as first:
        _LOGGER.debug("AI task failed (%s), trying once more", first)
        await asyncio.sleep(RETRY_AFTER)
        try:
            return await call(data)
        except HomeAssistantError:
            used = data.get("entity_id")
            others = [e for e in hass.states.async_entity_ids("ai_task") if e != used]
            if not others:
                raise first from None
            _LOGGER.info("AI task %s failed twice; %s answers this time", used or "(default)", others[0])
            return await call({**data, "entity_id": others[0]})


class AIUnavailable(HomeAssistantError):
    """The AI can't be used right now (today's limit, an error or a timeout)."""


# features with their own (optional) daily allowance; safety, the doorbell and spoken questions never have one
BUDGETS = {"drive": CONF_BUDGET_DRIVE, "noticing": CONF_BUDGET_NOTICING,
           "camera checks": CONF_BUDGET_CAMERAS, "watches": CONF_BUDGET_CAMERAS, "clip watch": CONF_BUDGET_CLIPS}
BUDGET_DEFAULTS = {CONF_BUDGET_CLIPS: DEFAULT_BUDGET_CLIPS}

class AIGate:
    def __init__(self, hass: HomeAssistant, entry_id: str, options: dict[str, Any]) -> None:
        self.hass = hass
        self.options = options
        self._store: Store[dict[str, Any]] = Store(hass, 1, f"{DOMAIN}.{entry_id}.ai")
        self.day = dt_util.now().date().isoformat()
        self.counts: dict[str, int] = {}
        self.history: dict[str, int] = {}
        self.failed_today = 0
        self.fails_in_row = 0
        self.last_error: str | None = None
        self.last_error_at: str | None = None
        self.last_ok_at: str | None = None
        self.capped_at: str | None = None

    async def async_start(self) -> None:
        data = await self._store.async_load() or {}
        self.history = dict(data.get("history") or {})
        self.fails_in_row = int(data.get("fails_in_row") or 0)
        self.last_error, self.last_error_at = data.get("last_error"), data.get("last_error_at")
        self.last_ok_at = data.get("last_ok_at")
        if data.get("day") == self.day:
            self.counts = dict(data.get("counts") or {})
            self.failed_today = int(data.get("failed_today") or 0)
            self.capped_at = data.get("capped_at")
        elif data.get("day") and data.get("counts"):
            self.history[data["day"]] = sum(data["counts"].values())
            self._trim()

    def _save(self) -> None:
        self._store.async_delay_save(lambda: {
            "day": self.day, "counts": self.counts, "history": self.history, "failed_today": self.failed_today,
            "fails_in_row": self.fails_in_row, "last_error": self.last_error, "last_error_at": self.last_error_at,
            "last_ok_at": self.last_ok_at, "capped_at": self.capped_at,
        }, 5)

    def _trim(self) -> None:
        for day in sorted(self.history)[:-HISTORY_DAYS]:
            self.history.pop(day, None)

    def _roll(self) -> None:
        today = dt_util.now().date().isoformat()
        if today == self.day:
            return
        if self.counts:
            self.history[self.day] = sum(self.counts.values())
            self._trim()
        self.day, self.counts, self.failed_today, self.capped_at = today, {}, 0, None
        self._save()

    # ----- the meter and the cap ----------------------------------------------------------
    @property
    def limit(self) -> int:
        return int(self.options.get(CONF_DAILY_AI_LIMIT, DEFAULT_DAILY_AI_LIMIT) or 0)

    @property
    def used(self) -> int:
        self._roll()
        return sum(self.counts.values())

    @property
    def capped(self) -> bool:
        return bool(self.limit) and self.used >= self.limit

    @property
    def failing(self) -> bool:
        return self.fails_in_row >= FAILING_AFTER

    def spend(self, purpose: str = "camera", force: bool = False) -> bool:
        """Count one background call. False (and nothing counted) once today's limit is used up —
        unless `force`, for something a person asked for, which is always allowed."""
        self._roll()
        if not force and self.over_budget(purpose):
            return False  # this feature's own allowance is used up; the rest carry on
        if not force and self.capped:
            if self.capped_at is None:
                self.capped_at = dt_util.utcnow().isoformat()
                _LOGGER.info("Sentient: today's AI limit (%s) is used up; plain rules until midnight", self.limit)
                self._save()
            return False
        self.counts[purpose] = self.counts.get(purpose, 0) + 1
        self._save()
        return True

    def over_budget(self, purpose: str) -> str | None:
        """This feature has used its own daily allowance (if it has one): the setting's name, else None."""
        key = BUDGETS.get(purpose)
        cap = int(float(self.options.get(key, BUDGET_DEFAULTS.get(key, 0)) or 0)) if key else 0
        if not cap:
            return None
        used = sum(n for p, n in self.counts.items() if BUDGETS.get(p) == key)
        return key if used >= cap else None

    def budgets(self) -> dict[str, dict[str, int]]:
        out = {}
        for key in dict.fromkeys(BUDGETS.values()):
            if cap := int(float(self.options.get(key, BUDGET_DEFAULTS.get(key, 0)) or 0)):
                out[key.removeprefix("budget_")] = {
                    "used": sum(n for p, n in self.counts.items() if BUDGETS.get(p) == key), "cap": cap}
        return out

    # ----- calling -------------------------------------------------------------------------
    async def async_call(self, data: dict[str, Any], timeout: float = TIMEOUT) -> dict[str, Any]:
        """ai_task.generate_data with a time limit; failures are noted and raised as AIUnavailable."""
        data = dict(data)
        if "entity_id" not in data and (ai := ai_task_entity(self.hass, self.options)):
            data["entity_id"] = ai
        try:
            async with asyncio.timeout(timeout):
                response = await _forgiving_call(self.hass, data)
        except Exception as err:  # noqa: BLE001 - every failure means "use the plain version"
            self.failed(err)
            raise AIUnavailable(f"The AI didn't answer: {err or type(err).__name__}") from err
        self.ok()
        return response or {}

    def failed(self, err: BaseException) -> None:
        self._roll()
        self.fails_in_row += 1
        self.failed_today += 1
        self.last_error = (str(err) or type(err).__name__)[:200]
        self.last_error_at = dt_util.utcnow().isoformat()
        _LOGGER.debug("Sentient AI call failed (%s in a row): %s", self.fails_in_row, self.last_error)
        self._save()

    def ok(self) -> None:
        was = self.fails_in_row
        self.fails_in_row = 0
        self.last_ok_at = dt_util.utcnow().isoformat()
        if was:
            self._save()

    def describe(self) -> dict[str, Any]:
        self._roll()
        days = sorted(self.history)[-7:]
        week = [self.history[d] for d in days]
        return {
            "today": self.used,
            "limit": self.limit,
            "capped": self.capped,
            "by_purpose": dict(sorted(self.counts.items(), key=lambda kv: -kv[1])),
            "budgets": self.budgets(),
            "failed_today": self.failed_today,
            "failing": self.failing,
            "last_error": self.last_error if self.failing else None,
            "last_ok": self.last_ok_at,
            "last_7_days": dict(zip(days, week)),
            "average_per_day": round(sum(week) / len(week)) if week else None,
        }


def gate(hass: HomeAssistant) -> AIGate | None:
    entries = hass.config_entries.async_loaded_entries(DOMAIN)
    rt = getattr(entries[0], "runtime_data", None) if entries else None
    return getattr(rt, "ai", None)


async def async_generate(hass: HomeAssistant, data: dict[str, Any], timeout: float = TIMEOUT) -> dict[str, Any]:
    """Call ai_task.generate_data through the gate (time limit, failure tracking)."""
    if (g := gate(hass)) is not None:
        return await g.async_call(data, timeout)
    try:
        async with asyncio.timeout(timeout):
            return await _forgiving_call(hass, data) or {}
    except Exception as err:  # noqa: BLE001
        raise AIUnavailable(f"The AI didn't answer: {err or type(err).__name__}") from err

