"""Which AI Task entity writes Sentient's background text (briefings, digests, camera looks...)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.core import HomeAssistant

from .const import CONF_AI_TASK_ENTITY

_LOGGER = logging.getLogger(__name__)
RETRY_AFTER = 2  # seconds


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
    first = next(iter(hass.states.async_entity_ids("ai_task")), None)
    return first


async def async_generate(hass: HomeAssistant, data: dict[str, Any]) -> Any:
    """ai_task.generate_data, a little more forgiving than calling it straight.

    A local model sometimes answers with nothing at all (busy, or it only "thought"), which Home
    Assistant reports as an error. Try once more; if that fails too and the house has another AI Task,
    let that one answer this time.
    """
    from homeassistant.exceptions import HomeAssistantError  # noqa: PLC0415

    async def call(payload: dict[str, Any]) -> Any:
        return await hass.services.async_call(
            "ai_task", "generate_data", payload, blocking=True, return_response=True
        )

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
