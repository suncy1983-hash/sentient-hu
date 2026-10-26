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


async def test_generate_retries_then_falls_back_to_another_ai_task(hass: HomeAssistant) -> None:
    from unittest.mock import patch

    from homeassistant.core import ServiceCall, SupportsResponse
    from homeassistant.exceptions import HomeAssistantError

    from custom_components.sentient.ai import async_generate

    hass.states.async_set("ai_task.local", "unknown")
    hass.states.async_set("ai_task.cloud", "unknown")
    seen: list[str] = []

    async def generate(call: ServiceCall):
        seen.append(call.data["entity_id"])
        if call.data["entity_id"] == "ai_task.local":
            raise HomeAssistantError("Last content in chat log is not an AssistantContent")
        return {"conversation_id": "x", "data": "ok"}

    hass.services.async_register("ai_task", "generate_data", generate, supports_response=SupportsResponse.ONLY)
    with patch("custom_components.sentient.ai.RETRY_AFTER", 0):
        res = await async_generate(hass, {"task_name": "t", "instructions": "i", "entity_id": "ai_task.local"})
    assert res["data"] == "ok"
    assert seen == ["ai_task.local", "ai_task.local", "ai_task.cloud"]
