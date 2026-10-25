"""Phone notifications: the important things also go to people's phones.

Driven by the timeline, so anything the house says, holds back, sees at the door or
finds wrong can be mirrored — each kind switched on separately. Uses notify entities
(e.g. the companion app), so it works with whatever the house already has.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import Any

from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from .const import CONF_NOTIFY_TARGETS, CONF_NOTIFY_WHEN, CONF_PERSONA, DEFAULT_NOTIFY_WHEN, DEFAULT_PERSONA
from . import i18n
from .timeline import EVENT_TIMELINE

_LOGGER = logging.getLogger(__name__)
_REPEAT = timedelta(minutes=10)


def wanted(item: dict[str, Any], kinds: set[str]) -> str | None:
    """The notification title for a timeline item, or None if it shouldn't be sent."""
    kind = item.get("kind")
    if kind == "critical" and "critical" in kinds:
        return "ALERT"
    if kind == "security" and "security" in kinds:
        return "Night watch" if item.get("alert") else "Left open"
    if kind == "said" and "urgent" in kinds and item.get("priority") == "urgent" and item.get("source") != "critical":
        return "Announcement"
    if kind == "held":
        reason = str(item.get("reason") or "")
        if "held_quiet" in kinds and reason == "quiet_hours":
            return "Held back for quiet hours"
        if "unheard" in kinds and reason == "no_speaker":
            return "Nobody heard this"
    if kind == "visitor" and "visitors" in kinds and not item.get("quiet"):
        return "Parcel left" if item.get("parcel") else "At the door"
    if kind == "watch" and "watches" in kinds:
        return "Watch"
    if kind == "digest" and "digest" in kinds:
        return "Daily digest"
    if kind == "learned" and "digest" in kinds:
        return "This week"
    if kind == "health" and "health" in kinds:
        return "House health"
    return None


class Notifier:
    """Sends chosen timeline items to notify entities."""

    def __init__(self, hass: HomeAssistant, options: dict[str, Any]) -> None:
        self.hass = hass
        self.options = options
        self._sent: dict[str, datetime] = {}
        self._unsub = None
        self.muted: set[str] = set()  # phones switched off with their "Notifications: <name>" switch

    @callback
    def async_start(self) -> None:
        self._unsub = self.hass.bus.async_listen(EVENT_TIMELINE, self._item)

    @callback
    def async_stop(self) -> None:
        if self._unsub:
            self._unsub()
            self._unsub = None

    def owner(self, target: str) -> str | None:
        """The person whose phone a notify entity is: the person tracked by a tracker on the same device."""
        reg = er.async_get(self.hass)
        entry = reg.async_get(target)
        if entry is None or not entry.device_id:
            return None
        trackers = {e.entity_id for e in er.async_entries_for_device(reg, entry.device_id) if e.domain == "device_tracker"}
        if not trackers:
            return None
        for person in self.hass.states.async_all("person"):
            if trackers & set(person.attributes.get("device_trackers") or []):
                return person.entity_id
        return None

    def at_home(self, target: str) -> bool:
        person = self.owner(target)
        st = self.hass.states.get(person) if person else None
        return st is not None and st.state == "home"

    @callback
    def _item(self, event: Event) -> None:
        targets = [
            t for t in self.options.get(CONF_NOTIFY_TARGETS) or []
            if self.hass.states.get(t) and t not in self.muted
        ]
        if not targets:
            return
        kinds = set(self.options.get(CONF_NOTIFY_WHEN, DEFAULT_NOTIFY_WHEN) or [])
        title = wanted(event.data, kinds)
        if title is None:
            return
        if event.data.get("kind") == "visitor":
            # whoever is at home hears it from the speakers; phones are for those who are out
            targets = [t for t in targets if not self.at_home(t)]
            if not targets:
                return
        text = event.data.get("text", "")
        now = dt_util.utcnow()
        self._sent = {k: v for k, v in self._sent.items() if now - v < _REPEAT}
        if text.lower() in self._sent:
            return
        self._sent[text.lower()] = now
        persona = self.options.get(CONF_PERSONA) or DEFAULT_PERSONA
        self.hass.async_create_task(self._send(targets, f"{persona} · {i18n.title(title)}", text), eager_start=True)

    async def _send(self, targets: list[str], title: str, message: str) -> None:
        try:
            await self.hass.services.async_call(
                "notify", "send_message", {"entity_id": targets, "title": title, "message": message}, blocking=True
            )
        except Exception as err:  # noqa: BLE001 - a failed notification must never break anything else
            _LOGGER.warning("Sentient couldn't send a notification: %s", err)
