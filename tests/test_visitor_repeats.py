"""The same person walking past again isn't announced again; phones only for who is out; a switch per phone."""

from unittest.mock import patch

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.helpers import device_registry as dr, entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

from custom_components.sentient.const import DOMAIN

OPTIONS = {
    "persona": "Nova",
    "speakers": ["assist_satellite.kitchen"],
    "quiet_start": "00:00:00",
    "quiet_end": "00:00:00",
    "door_camera": "camera.door",
    "door_triggers": ["binary_sensor.door_person"],
    "face_sensor": "sensor.door_face",
    "announce_visitors": True,
    "parcel_alerts": False,
    "notify_targets": ["notify.sanya_phone", "notify.kata_phone"],
    "notify_when": ["visitors", "critical"],
}


def _phone(hass: HomeAssistant, who: str) -> None:
    """A companion-app phone: a notify entity and a tracker on one device, the tracker on a person."""
    owner = MockConfigEntry(domain="mobile_app")
    owner.add_to_hass(hass)
    device = dr.async_get(hass).async_get_or_create(config_entry_id=owner.entry_id, identifiers={("mobile_app", who)})
    reg = er.async_get(hass)
    reg.async_get_or_create("notify", "mobile_app", f"{who}_n", suggested_object_id=f"{who}_phone",
                            config_entry=owner, device_id=device.id)
    reg.async_get_or_create("device_tracker", "mobile_app", f"{who}_t", suggested_object_id=f"{who}_phone",
                            config_entry=owner, device_id=device.id)
    hass.states.async_set(f"notify.{who}_phone", "unknown")


async def _setup(hass: HomeAssistant, replies: list[dict]) -> tuple[MockConfigEntry, list[ServiceCall]]:
    _phone(hass, "sanya")
    _phone(hass, "kata")
    hass.states.async_set("person.sanya", "home", {"friendly_name": "Sanya", "device_trackers": ["device_tracker.sanya_phone"]})
    hass.states.async_set("person.kata", "not_home", {"friendly_name": "Kata", "device_trackers": ["device_tracker.kata_phone"]})
    entry = MockConfigEntry(domain=DOMAIN, title="Nova", data=OPTIONS, unique_id=DOMAIN)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    hass.states.async_set("assist_satellite.kitchen", "idle")
    hass.states.async_set("camera.door", "idle")
    hass.states.async_set("binary_sensor.door_person", "off")
    hass.states.async_set("sensor.door_face", "unknown")
    await hass.async_block_till_done()
    calls: list[ServiceCall] = []

    async def generate(call: ServiceCall):
        calls.append(call)
        return {"conversation_id": "x", "data": replies[min(len(calls), len(replies)) - 1]}

    hass.services.async_register("ai_task", "generate_data", generate, supports_response=SupportsResponse.ONLY)
    return entry, calls


async def _walk_past(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    entry.runtime_data.visitors._last = None  # past the one-minute cooldown
    with patch("custom_components.sentient.visitors.asyncio.sleep"):
        hass.states.async_set("binary_sensor.door_person", "on")
        await hass.async_block_till_done()
        hass.states.async_set("binary_sensor.door_person", "off")
        await hass.async_block_till_done()


async def test_the_same_clothes_again_are_not_announced_again(hass: HomeAssistant) -> None:
    entry, ai = await _setup(hass, [
        {"kind": "visitor", "description": "A man in a brown jumper", "clothes": "brown jumper, glasses", "same_as": 0},
        {"kind": "visitor", "description": "A man with a phone", "clothes": "brown jumper, glasses", "same_as": 1},
        {"kind": "visitor", "description": "A man walking", "clothes": "brown jumper", "same_as": 1},
        {"kind": "delivery", "description": "A courier", "clothes": "yellow vest", "same_as": 0},
    ])
    said = async_mock_service(hass, "assist_satellite", "announce")
    sent = async_mock_service(hass, "notify", "send_message")
    for _ in range(3):
        await _walk_past(hass, entry)
    assert len(said) == 1 and len(sent) == 1  # walked past three times, told once
    assert "brown jumper, glasses" in ai[1].data["instructions"]  # it was shown who it saw earlier
    log = entry.runtime_data.visitors.log
    assert [v["repeat"] for v in log] == [True, True, False]

    await _walk_past(hass, entry)  # someone new
    assert len(said) == 2 and len(sent) == 2


async def test_clothes_carry_a_recognised_name(hass: HomeAssistant) -> None:
    entry, _ai = await _setup(hass, [
        {"kind": "visitor", "description": "A man in a brown jumper", "clothes": "brown jumper", "same_as": 0},
        {"kind": "visitor", "description": "A man from behind", "clothes": "brown jumper", "same_as": 1},
    ])
    said = async_mock_service(hass, "assist_satellite", "announce")
    hass.states.async_set("sensor.door_face", "Sanya")  # the camera knew his face the first time
    await hass.async_block_till_done()
    await _walk_past(hass, entry)
    hass.states.async_set("sensor.door_face", "Unknown")  # not the second time
    await hass.async_block_till_done()
    await _walk_past(hass, entry)
    first, second = entry.runtime_data.visitors.log[1], entry.runtime_data.visitors.log[0]
    assert first["name"] == "Sanya" and first["household"]
    assert second["name"] == "Sanya" and second["household"]
    assert not said


async def test_phones_only_for_who_is_out_and_a_switch_per_phone(hass: HomeAssistant) -> None:
    entry, _ai = await _setup(hass, [{"kind": "visitor", "description": "A stranger", "clothes": "red coat", "same_as": 0}])
    async_mock_service(hass, "assist_satellite", "announce")
    sent = async_mock_service(hass, "notify", "send_message")
    await _walk_past(hass, entry)
    assert sent[-1].data["entity_id"] == ["notify.kata_phone"]  # Sanya is at home and hears the speaker

    switches = {s.attributes.get("notify_entity"): s.entity_id for s in hass.states.async_all("switch")
                if s.attributes.get("notify_entity")}
    assert set(switches) == {"notify.sanya_phone", "notify.kata_phone"}
    kata_switch = hass.states.get(switches["notify.kata_phone"])
    assert kata_switch.state == "on" and kata_switch.attributes["person"] == "person.kata"
    assert kata_switch.name.endswith("Kata")

    await hass.services.async_call("switch", "turn_off", {"entity_id": switches["notify.kata_phone"]}, blocking=True)
    hass.bus.async_fire("sentient_timeline", {"kind": "critical", "text": "Water leak under the sink"})
    await hass.async_block_till_done()
    assert sent[-1].data["entity_id"] == ["notify.sanya_phone"]  # critical goes to everyone switched on


async def test_a_visible_person_is_never_nobody(hass: HomeAssistant) -> None:
    entry, _ai = await _setup(hass, [
        {"person": True, "kind": "nobody", "description": "Someone running past the gate", "clothes": "white top"},
    ])
    said = async_mock_service(hass, "assist_satellite", "announce")
    await _walk_past(hass, entry)
    assert entry.runtime_data.visitors.log[0]["kind"] == "unknown"
    assert len(said) == 1


async def test_announce_visitors_switch_silences_the_speakers_only(hass: HomeAssistant) -> None:
    entry, _ai = await _setup(hass, [{"kind": "visitor", "description": "A stranger", "clothes": "red coat", "same_as": 0}])
    said = async_mock_service(hass, "assist_satellite", "announce")
    sent = async_mock_service(hass, "notify", "send_message")
    switch = next(s.entity_id for s in hass.states.async_all("switch") if s.entity_id.endswith("announce_visitors"))
    assert hass.states.get(switch).state == "on"
    await hass.services.async_call("switch", "turn_off", {"entity_id": switch}, blocking=True)
    await hass.async_block_till_done()
    assert hass.states.get(switch).state == "off"
    await _walk_past(hass, entry)
    assert not said  # the speakers stay quiet
    assert len(sent) == 1  # Kata, who is out, still gets it on her phone
