"""Which AI Task entity writes Sentient's background text (briefings, digests, camera looks...)."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from .const import CONF_AI_TASK_ENTITY


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
