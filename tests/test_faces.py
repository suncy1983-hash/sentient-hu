"""Sentient's own face recognition at the door: used when the camera can't name anyone, and taught by hand."""

from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

from custom_components.sentient.const import DOMAIN

OPTIONS = {
    "persona": "Nova",
    "speakers": ["assist_satellite.kitchen"],
    "quiet_start": "00:00:00",
    "quiet_end": "00:00:00",
    "door_camera": "camera.door",
    "door_triggers": ["binary_sensor.door_person"],
    "announce_visitors": True,
    "parcel_alerts": False,
    "voice_id_url": "http://voice-id:8765",
}


async def _setup(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, title="Nova", data=OPTIONS, unique_id=DOMAIN)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    hass.states.async_set("assist_satellite.kitchen", "idle")
    hass.states.async_set("camera.door", "idle", {"friendly_name": "Door"})
    hass.states.async_set("binary_sensor.door_person", "off")
    hass.states.async_set("person.kata", "home", {"friendly_name": "Kata"})
    await hass.async_block_till_done()

    async def generate(call: ServiceCall):
        return {"conversation_id": "x", "data": {"kind": "visitor", "description": "A woman in a green coat"}}

    hass.services.async_register("ai_task", "generate_data", generate, supports_response=SupportsResponse.ONLY)
    return entry


async def _ring(hass: HomeAssistant) -> None:
    with patch("custom_components.sentient.visitors.asyncio.sleep"):
        hass.states.async_set("binary_sensor.door_person", "on")
        await hass.async_block_till_done()
        hass.states.async_set("binary_sensor.door_person", "off")
        await hass.async_block_till_done()


async def test_own_face_recognition_names_the_family_and_stays_quiet(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    said = async_mock_service(hass, "assist_satellite", "announce")
    faces = entry.runtime_data.faces
    with patch("custom_components.sentient.visitors.async_picture", AsyncMock(return_value=b"jpeg")), \
         patch.object(faces, "async_identify", AsyncMock(return_value={"name": "Kata", "id": "f1", "count": 1})) as ident:
        await _ring(hass)
    ident.assert_awaited_once_with(b"jpeg")
    visit = entry.runtime_data.visitors.log[0]
    assert visit["name"] == "Kata" and visit["household"] and visit["face_id"] == "f1"
    assert not said


async def test_a_stranger_is_announced_and_can_be_taught_afterwards(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    said = async_mock_service(hass, "assist_satellite", "announce")
    faces = entry.runtime_data.faces
    with patch("custom_components.sentient.visitors.async_picture", AsyncMock(return_value=b"jpeg")), \
         patch.object(faces, "async_identify", AsyncMock(return_value={"name": None, "id": "f2", "count": 1})):
        await _ring(hass)
    assert said[-1].data["message"] == "Someone's at the door: A woman in a green coat."

    # "that was Kata": taught from the face it has just seen
    ctx = llm.LLMContext(platform="test", context=None, language="en", assistant="conversation", device_id=None)
    api = await llm.async_get_api(hass, "sentient", ctx)
    with patch.object(faces, "async_learn_ids", AsyncMock(return_value=1)) as learn:
        out = await api.async_call_tool(llm.ToolInput(tool_name="sentient_learn_face", tool_args={"name": "Kata"}))
    learn.assert_awaited_once_with("Kata", ["f2"])
    assert out["learned"] == "Kata" and out["samples"] == 1
    visit = entry.runtime_data.visitors.log[0]
    assert visit["name"] == "Kata" and visit["household"]


async def test_learning_a_face_from_a_camera_and_a_photo(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    faces = entry.runtime_data.faces
    ctx = llm.LLMContext(platform="test", context=None, language="en", assistant="conversation", device_id=None)
    api = await llm.async_get_api(hass, "sentient", ctx)
    with patch("custom_components.sentient.visitors.async_picture", AsyncMock(return_value=b"jpeg")) as pic, \
         patch.object(faces, "async_learn_picture", AsyncMock(return_value=2)) as learn:
        out = await api.async_call_tool(
            llm.ToolInput(tool_name="sentient_learn_face", tool_args={"name": "Szofi", "camera": "Door"})
        )
    pic.assert_awaited_once_with(hass, "camera.door")
    learn.assert_awaited_once_with("Szofi", b"jpeg")
    assert out == {"learned": "Szofi", "from": "Door", "samples": 2}

    # no visit with a face yet: a clear answer, not a crash
    out = await api.async_call_tool(llm.ToolInput(tool_name="sentient_learn_face", tool_args={"name": "Kata"}))
    assert "no recent visit" in out["error"]

    with patch("custom_components.sentient.faces.async_photo", AsyncMock(return_value=b"photo")) as photo, \
         patch.object(faces, "async_learn_picture", AsyncMock(return_value=1)) as learn:
        res = await hass.services.async_call(
            DOMAIN, "learn_face",
            {"name": "Sanya", "photo": {"media_content_id": "media-source://media_source/local/sanya.jpg",
                                       "media_content_type": "image/jpeg"}},
            blocking=True, return_response=True,
        )
    photo.assert_awaited_once()
    assert photo.await_args.args[1] == "media-source://media_source/local/sanya.jpg"
    learn.assert_awaited_once_with("Sanya", b"photo")
    assert res == {"learned": "Sanya", "samples": 1}


async def test_face_tools_only_with_the_voice_id_app(hass: HomeAssistant) -> None:
    await _setup(hass)
    ctx = llm.LLMContext(platform="test", context=None, language="en", assistant="conversation", device_id=None)
    api = await llm.async_get_api(hass, "sentient", ctx)
    names = {t.name for t in api.tools}
    assert {"sentient_learn_face", "sentient_faces", "sentient_forget_face"} <= names
