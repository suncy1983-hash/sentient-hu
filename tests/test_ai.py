"""Choosing the AI Task entity when Sentient's option is empty."""

from homeassistant.core import HomeAssistant

from custom_components.sentient.ai import ai_task_entity
from custom_components.sentient.const import CONF_AI_TASK_ENTITY


async def test_configured_entity_wins(hass: HomeAssistant) -> None:
    hass.states.async_set("ai_task.other", "unknown")
    assert ai_task_entity(hass, {CONF_AI_TASK_ENTITY: "ai_task.mine"}) == "ai_task.mine"


async def test_falls_back_to_an_existing_entity(hass: HomeAssistant) -> None:
    hass.states.async_set("ai_task.google_ai_task", "unknown")
    assert ai_task_entity(hass, {}) == "ai_task.google_ai_task"


async def test_nothing_available(hass: HomeAssistant) -> None:
    assert ai_task_entity(hass, {}) is None
