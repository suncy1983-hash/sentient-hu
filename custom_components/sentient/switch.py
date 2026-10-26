"""Switches: hush the house for a while, guest mode, the chime and disco mode."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from . import SentientConfigEntry
from .const import (
    CONF_ANNOUNCE_VISITORS,
    CONF_CHIME,
    CONF_NOTIFY_TARGETS,
    DEFAULT_ANNOUNCE_VISITORS,
    DOMAIN,
    EVENT_MODE_CHANGED,
)

DEFAULT_HUSH_MINUTES = 60


async def async_setup_entry(
    hass: HomeAssistant, entry: SentientConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    phones = [NotifyTargetSwitch(hass, entry, target) for target in entry.runtime_data.options.get(CONF_NOTIFY_TARGETS) or []]
    async_add_entities([HushSwitch(entry), GuestSwitch(entry), ChimeSwitch(entry), PartySwitch(entry), AnnounceVisitorsSwitch(entry), *phones])


class NotifyTargetSwitch(SwitchEntity, RestoreEntity):
    """One per phone Sentient notifies: off, and that phone gets nothing from Sentient."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_icon = "mdi:cellphone-message"
    _attr_translation_key = "notify_target"

    def __init__(self, hass: HomeAssistant, entry: SentientConfigEntry, target: str) -> None:
        self._entry = entry
        self._target = target
        self._attr_unique_id = f"{entry.entry_id}_notify_{target}"
        # named after the phone's owner (person), else the phone itself
        person = entry.runtime_data.notifier.owner(target)
        named = hass.states.get(person) if person else hass.states.get(target)
        self._attr_translation_placeholders = {
            "name": named.name if named else target.split(".", 1)[-1].replace("_", " ").title()
        }
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)}, name=entry.title, manufacturer="Sentient",
            model="Home brain", entry_type=DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        last = await self.async_get_last_state()
        if last is not None and last.state == "off":
            self._entry.runtime_data.notifier.muted.add(self._target)

    @property
    def is_on(self) -> bool:
        return self._target not in self._entry.runtime_data.notifier.muted

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"notify_entity": self._target, "person": self._entry.runtime_data.notifier.owner(self._target)}

    async def async_turn_on(self, **kwargs: Any) -> None:
        self._entry.runtime_data.notifier.muted.discard(self._target)
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        self._entry.runtime_data.notifier.muted.add(self._target)
        self.async_write_ha_state()


class _ModeSwitch(SwitchEntity, RestoreEntity):
    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(self, entry: SentientConfigEntry, key: str) -> None:
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)}, name=entry.title, manufacturer="Sentient",
            model="Home brain", entry_type=DeviceEntryType.SERVICE,
        )

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(self.hass.bus.async_listen(EVENT_MODE_CHANGED, self._changed))

    @callback
    def _changed(self, _event: Event) -> None:
        self.async_write_ha_state()


class HushSwitch(_ModeSwitch):
    """On: the house says nothing that isn't urgent until the time runs out."""

    _attr_icon = "mdi:volume-off"

    def __init__(self, entry: SentientConfigEntry) -> None:
        super().__init__(entry, "hush")
        self._unsub_end = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last and last.state == "on" and (until := dt_util.parse_datetime(last.attributes.get("until") or "")):
            if until > dt_util.utcnow():
                hush(self.hass, self._entry, until)

    @property
    def is_on(self) -> bool:
        return self._entry.runtime_data.speech.hushed()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        until = self._entry.runtime_data.speech.hush_until
        return {"until": until.isoformat() if until and self.is_on else None}

    async def async_turn_on(self, **kwargs: Any) -> None:
        hush(self.hass, self._entry, dt_util.utcnow() + timedelta(minutes=DEFAULT_HUSH_MINUTES))

    async def async_turn_off(self, **kwargs: Any) -> None:
        hush(self.hass, self._entry, None)


class GuestSwitch(_ModeSwitch):
    """On: guests are in — no remarks or routine offers, and personal memories stay private."""

    _attr_icon = "mdi:account-multiple"

    def __init__(self, entry: SentientConfigEntry) -> None:
        super().__init__(entry, "guest_mode")

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last and last.state == "on":
            self._entry.runtime_data.speech.guest_mode = True

    @property
    def is_on(self) -> bool:
        return self._entry.runtime_data.speech.guest_mode

    async def async_turn_on(self, **kwargs: Any) -> None:
        set_guest_mode(self.hass, self._entry, True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        set_guest_mode(self.hass, self._entry, False)


class ChimeSwitch(_ModeSwitch):
    """The speakers' chime before Sentient speaks (the same as the setting)."""

    _attr_icon = "mdi:bell-ring-outline"

    def __init__(self, entry: SentientConfigEntry) -> None:
        super().__init__(entry, "chime")

    @property
    def is_on(self) -> bool:
        return self._entry.runtime_data.speech.chime

    async def async_turn_on(self, **kwargs: Any) -> None:
        self._set(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        self._set(False)

    def _set(self, on: bool) -> None:
        entry = self._entry
        self.hass.config_entries.async_update_entry(entry, options={**entry.data, **entry.options, CONF_CHIME: on})
        entry.runtime_data.speech.options = {**entry.runtime_data.speech.options, CONF_CHIME: on}
        self.async_write_ha_state()
        self.hass.bus.async_fire(EVENT_MODE_CHANGED, {"chime": on})


class AnnounceVisitorsSwitch(_ModeSwitch):
    """Whether the speakers say who has come to the door or gate (the visitor log and phones carry on)."""

    _attr_icon = "mdi:account-voice"

    def __init__(self, entry: SentientConfigEntry) -> None:
        super().__init__(entry, "announce_visitors")

    @property
    def is_on(self) -> bool:
        return bool(self._entry.runtime_data.visitors.options.get(CONF_ANNOUNCE_VISITORS, DEFAULT_ANNOUNCE_VISITORS))

    async def async_turn_on(self, **kwargs: Any) -> None:
        self._set(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        self._set(False)

    def _set(self, on: bool) -> None:
        entry = self._entry
        self.hass.config_entries.async_update_entry(
            entry, options={**entry.data, **entry.options, CONF_ANNOUNCE_VISITORS: on}
        )
        visitors = entry.runtime_data.visitors
        visitors.options = {**visitors.options, CONF_ANNOUNCE_VISITORS: on}
        self.async_write_ha_state()
class PartySwitch(_ModeSwitch):
    """Light show: the chosen lights put on a show, then go back as they were."""

    _attr_icon = "mdi:party-popper"

    def __init__(self, entry: SentientConfigEntry) -> None:
        super().__init__(entry, "party_mode")

    @property
    def is_on(self) -> bool:
        return self._entry.runtime_data.party.active

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        party = self._entry.runtime_data.party
        return {
            "style": party.style if party.active else None,
            "until": party.until.isoformat() if party.active and party.until else None,
            "lights": list(party.lights) if party.active else [],
            "following_sound": party.listening,
            "buttons": list(party.options.get("party_show", ["disco"]) or []),
        }

    async def async_turn_on(self, **kwargs: Any) -> None:
        result = await self._entry.runtime_data.party.async_party()
        if result.get("error"):
            from homeassistant.exceptions import HomeAssistantError

            raise HomeAssistantError(result["error"])

    async def async_turn_off(self, **kwargs: Any) -> None:
        await self._entry.runtime_data.party.async_end()


_HUSH_TIMERS: dict[str, Any] = {}


@callback
def hush(hass: HomeAssistant, entry, until) -> None:
    """Hush until a time (None stops it) and refresh the switch when it ends."""
    speech = entry.runtime_data.speech
    speech.hush_until = until
    if until is not None and (fb := getattr(entry.runtime_data, "feedback", None)) is not None:
        if ids := fb.hushed():
            names = [st.name if (st := hass.states.get(e)) else e for e in ids]
            entry.runtime_data.timeline.add("mode", "Noted: hushed straight after a remark", ", ".join(names))
    if (unsub := _HUSH_TIMERS.pop(entry.entry_id, None)) is not None:
        unsub()
    if until is not None:
        @callback
        def _end(_now) -> None:
            _HUSH_TIMERS.pop(entry.entry_id, None)
            hass.bus.async_fire(EVENT_MODE_CHANGED, {"hush": False})

        _HUSH_TIMERS[entry.entry_id] = async_track_point_in_utc_time(hass, _end, until)
    hass.bus.async_fire(EVENT_MODE_CHANGED, {"hush": until is not None, "until": until.isoformat() if until else None})


@callback
def set_guest_mode(hass: HomeAssistant, entry, on: bool) -> None:
    entry.runtime_data.speech.guest_mode = on
    hass.bus.async_fire(EVENT_MODE_CHANGED, {"guest_mode": on})
