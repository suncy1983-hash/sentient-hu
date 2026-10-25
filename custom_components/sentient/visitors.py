"""Visitors: a quiet log of who came to the door.

When a chosen doorbell (or any trigger the user picks) fires, Sentient takes
one AI look through the door camera and logs the visit — "2:02pm, a delivery
driver holding a small parcel". By default it says nothing about who is
there; voice agents can answer "who came earlier?" from the log. If the
visitor looked like a delivery, it looks again a few minutes later and can
say that a parcel has been left.

Optional doorstep sensors (e.g. a Frigate zone's person occupancy) catch the
deliveries that don't ring: when a visit there ends and the bell wasn't
pressed, one look checks for a parcel or post left behind. It only speaks
when a parcel appears that wasn't there at the last look.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.const import EVENT_STATE_CHANGED, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .ai import ai_task_entity
from .const import (
    CONF_ANNOUNCE_VISITORS,
    CONF_DOOR_CAMERA,
    CONF_DOOR_TRIGGERS,
    CONF_DOORSTEP_SENSORS,
    CONF_FACE_SENSOR,
    CONF_PARCEL_ALERTS,
    DEFAULT_ANNOUNCE_VISITORS,
    DEFAULT_PARCEL_ALERTS,
    DOMAIN,
    EVENT_VISITOR,
)
from . import i18n
from .i18n import t
from .faces import FaceError, FaceID, async_picture
from .memory import _fold
from .watches import local_when

if TYPE_CHECKING:
    from .speech import SpeechRouter
    from .watches import WatchManager

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
COOLDOWN = timedelta(seconds=60)
PARCEL_RECHECK = 180  # seconds after a delivery
KEEP = 30
DOORSTEP_SETTLE = 10  # seconds after a doorstep visit ends, so the visitor has stepped away
DOORSTEP_COOLDOWN = timedelta(minutes=2)
RANG_RECENTLY = timedelta(minutes=5)  # a doorbell visit already does its own parcel check
FACE_FRESH = timedelta(minutes=2)
_NO_NAME = {"", "none", "unknown", "unavailable", "no face", "no_face", "nobody", "null",
            "stranger", "unknown person", "unrecognized", "unrecognised", "ismeretlen", "idegen"}
_FACE_WORDS = ("face", "last_person", "recogni")
HOUSEHOLD_MOVING = timedelta(minutes=3)
EVENT_IMAGE_WAIT = 20  # seconds to wait for a battery camera's event snapshot
EVENT_IMAGE_SLACK = timedelta(seconds=10)
KINDS = ["delivery", "visitor", "neighbour", "child", "unknown", "nobody"]


def _media_id(entity_id: str | None) -> str:
    """A camera's live still or an image entity, as a media source."""
    domain = (entity_id or "camera.").split(".", 1)[0]
    return f"media-source://{'image' if domain == 'image' else 'camera'}/{entity_id}"


class Visitors:
    """Watches the door triggers and keeps the log."""

    def __init__(
        self, hass: HomeAssistant, entry_id: str, options: dict[str, Any], speech: SpeechRouter, watches: WatchManager
    ) -> None:
        self.hass = hass
        self.options = options
        self.speech = speech
        self.watches = watches
        self._store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}.visitors")
        self.log: deque[dict[str, Any]] = deque(maxlen=KEEP)
        self._last: datetime | None = None
        self._busy = False
        self._unsubs: list = []
        self._doorstep_last: datetime | None = None
        self._doorstep_pending = False
        self.parcel_waiting = False  # a parcel was there at the last look
        self._person_moved: datetime | None = None  # when someone who lives here last arrived or left
        self.faces: FaceID | None = None  # Sentient's own face recognition, when the Voice ID app is set up

    async def async_start(self) -> None:
        data = await self._store.async_load() or {}
        self.log.extend(data.get("visits", [])[:KEEP])
        self.parcel_waiting = bool(data.get("parcel_waiting"))
        self._unsubs.append(self.hass.bus.async_listen(EVENT_STATE_CHANGED, self._on_state_changed))

    @callback
    def async_stop(self) -> None:
        while self._unsubs:
            self._unsubs.pop()()

    @callback
    def _save(self) -> None:
        self._store.async_delay_save(lambda: {"visits": list(self.log), "parcel_waiting": self.parcel_waiting}, 2)
        self.hass.bus.async_fire(EVENT_VISITOR, {"count": len(self.log)})

    @property
    def camera(self) -> str | None:
        return self.options.get(CONF_DOOR_CAMERA)

    def _device_entities(self, entity_id: str | None, domain: str) -> list[str]:
        """Entities of one domain on the same device as entity_id."""
        if not entity_id:
            return []
        reg = er.async_get(self.hass)
        entry = reg.async_get(entity_id)
        if entry is None or not entry.device_id:
            return []
        return sorted(
            e.entity_id for e in er.async_entries_for_device(reg, entry.device_id)
            if e.domain == domain and not e.disabled_by
        )

    def camera_for(self, trigger: str | None) -> str | None:
        """The camera on the trigger's own device (a gate camera's person sensor looks through
        the gate camera), else the configured door camera."""
        own = self._device_entities(trigger, "camera")
        return own[0] if own else self.camera

    async def picture_for(self, trigger: str | None, camera: str | None) -> str | None:
        """What to look at for a visit. Battery cameras (eufy and the like) often have no live
        still while asleep, but their integration publishes an event snapshot as an image entity
        on the same device moments after a detection: wait briefly for that, else use the camera."""
        images = sorted(self._device_entities(trigger, "image"), key=lambda e: "event" not in e)
        if not images:
            await asyncio.sleep(1)
            return camera
        start = dt_util.utcnow() - EVENT_IMAGE_SLACK
        for _ in range(EVENT_IMAGE_WAIT):
            for image in images:
                st = self.hass.states.get(image)
                if st and st.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE) and st.last_changed >= start:
                    return image
            await asyncio.sleep(1)
        return camera

    async def _faces_in(self, picture: str | None) -> dict[str, Any] | None:
        """Who Sentient's own face recognition sees in the visit's picture (None if it can't tell)."""
        if not picture or self.faces is None:
            return None
        try:
            content = await async_picture(self.hass, picture)
        except HomeAssistantError as err:
            _LOGGER.debug("No picture for the face check: %s", err)
            return None
        return await self.faces.async_identify(content)

    async def async_learn_face(self, name: str, camera: str | None = None, visit: int = 0) -> dict[str, Any]:
        """Teach a face: from a camera right now, or from a recent visit ("that was Kata")."""
        if self.faces is None or not self.faces.enabled:
            raise FaceError("face recognition isn't set up (no Voice ID address in Sentient's settings)")
        name = name.strip()
        if camera:
            content = await async_picture(self.hass, camera)
            samples = await self.faces.async_learn_picture(name, content)
            return {"learned": name, "from": self.hass.states.get(camera).name if self.hass.states.get(camera) else camera,
                    "samples": samples}
        with_face = [v for v in self.log if v.get("face_id")]
        if visit >= len(with_face):
            raise FaceError("no recent visit with a clear face to learn from")
        chosen = with_face[visit]
        samples = await self.faces.async_learn_ids(name, [chosen["face_id"]])
        chosen["name"] = name
        chosen["household"] = self.is_household(name)
        self._save()
        return {"learned": name, "from_visit": chosen["time"], "description": chosen["description"], "samples": samples}

    def place_for(self, camera: str | None) -> str | None:
        """A name for where a camera looks, when it isn't the main door camera."""
        if not camera or camera == self.camera:
            return None
        st = self.hass.states.get(camera)
        return st.name if st else None

    def is_household(self, name: str | None) -> bool:
        """Whether a recognised name is one of the people who live here (HA person entities)."""
        if not name:
            return False
        wanted = _fold(name)
        reg = er.async_get(self.hass)
        for st in self.hass.states.async_all("person"):
            names = {_fold(st.name), _fold(st.entity_id.split(".", 1)[1].replace("_", " "))}
            if (entry := reg.async_get(st.entity_id)) is not None:
                names |= {_fold(alias) for alias in entry.aliases if isinstance(alias, str)}
            if wanted in names or wanted.split(" ")[0] in {n.split(" ")[0] for n in names if n}:
                return True
        return False

    def person_for_face(self, face_id: Any) -> str | None:
        """The household member given the alias 'face <id>' for a camera's face-recognition id."""
        wanted = f"face {face_id}".lower()
        reg = er.async_get(self.hass)
        for st in self.hass.states.async_all("person"):
            entry = reg.async_get(st.entity_id)
            if entry and any(isinstance(a, str) and a.strip().lower() == wanted for a in entry.aliases):
                return st.name
        return None

    def household_moving(self) -> bool:
        """Someone who lives here has just arrived or left — the camera most likely saw them."""
        return self._person_moved is not None and dt_util.utcnow() - self._person_moved < HOUSEHOLD_MOVING

    @property
    def triggers(self) -> set[str]:
        return set(self.options.get(CONF_DOOR_TRIGGERS) or [])

    @property
    def doorstep(self) -> set[str]:
        return set(self.options.get(CONF_DOORSTEP_SENSORS) or [])

    def today(self) -> list[dict[str, Any]]:
        start = dt_util.start_of_local_day()
        return [v for v in self.log if (t := dt_util.parse_datetime(v["time"])) and t >= start]

    def summary(self, limit: int = 8) -> list[str]:
        out = []
        for v in list(self.log)[:limit]:
            when = dt_util.parse_datetime(v["time"])
            who = f"{v['name']} — " if v.get("name") else ""
            line = f"{local_when(when) if when else ''}: {who}{v['description']}"
            if v.get("parcel_left"):
                line += " (left a parcel)"
            out.append(line)
        return out

    def recognised_name(self, trigger: str | None = None) -> str | None:
        """A name from a face-recognition sensor, if one appeared just now: a face/person-name
        sensor on the trigger's own device first, then the configured one."""
        candidates = [
            e for e in self._device_entities(trigger, "sensor")
            if any(word in e.split(".", 1)[1] for word in _FACE_WORDS)
        ]
        if configured := self.options.get(CONF_FACE_SENSOR):
            candidates.append(configured)
        for entity_id in candidates:
            st = self.hass.states.get(entity_id)
            if st is None:
                continue
            if st.state.strip().lower() in _NO_NAME:
                # the camera knew the face but the integration couldn't name it: a person whose
                # aliases include "face <id>" is who it was
                face_id = st.attributes.get("person_id")
                if face_id is not None and dt_util.utcnow() - st.last_updated <= FACE_FRESH:
                    if name := self.person_for_face(face_id):
                        return name
                continue
            if dt_util.utcnow() - st.last_changed > FACE_FRESH:
                continue
            try:
                float(st.state)
                continue  # a count or a score, not a name
            except ValueError:
                pass
            return st.state.strip().replace("_", " ").title()
        return None

    # ----- trigger -------------------------------------------------------------
    @callback
    def _on_state_changed(self, event: Event[EventStateChangedData]) -> None:
        new = event.data["new_state"]
        if new is not None and new.domain == "person":
            # a real arrival or departure, not a restart or a person entity coming back online
            was = event.data["old_state"]
            if was is not None and was.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN) \
                    and new.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN) and was.state != new.state:
                self._person_moved = dt_util.utcnow()
            return
        if new is None or (new.entity_id not in self.doorstep and new.entity_id not in self.triggers):
            return
        old = event.data["old_state"]
        if new.state in (STATE_UNAVAILABLE, STATE_UNKNOWN) or old is None or old.state == STATE_UNAVAILABLE:
            return
        if new.entity_id in self.doorstep:
            if self.camera and old.state == STATE_ON and new.state != STATE_ON:
                self._doorstep_visit_ended()
            return
        if not self.camera_for(new.entity_id):
            return
        if new.domain in ("event", "image"):
            # event entities change state to the time they fired; an image entity (a battery
            # camera's "last event" snapshot) to the time a new picture arrived
            fired = old.state != new.state and not (new.domain == "image" and old.state == STATE_UNKNOWN)
        else:
            fired = new.state == STATE_ON and old.state != STATE_ON
        if not fired:
            return
        now = dt_util.utcnow()
        if self._busy or (self._last and now - self._last < COOLDOWN):
            return
        self._last = now
        self.hass.async_create_task(self.async_visit(new.entity_id))

    # ----- doorstep visits without the bell -------------------------------------
    @callback
    def _doorstep_visit_ended(self) -> None:
        if not self.options.get(CONF_PARCEL_ALERTS, DEFAULT_PARCEL_ALERTS) or self._doorstep_pending:
            return
        now = dt_util.utcnow()
        if self._doorstep_last and now - self._doorstep_last < DOORSTEP_COOLDOWN:
            return
        if self._last and now - self._last < RANG_RECENTLY:
            return  # the bell rang: that visit has its own parcel check
        if any((st := self.hass.states.get(e)) and st.state == STATE_ON for e in self.doorstep):
            return  # someone's still there
        self._doorstep_pending = True

        @callback
        def _go(_now: datetime) -> None:
            self.hass.async_create_task(self.async_doorstep_check())

        async_call_later(self.hass, DOORSTEP_SETTLE, _go)

    async def async_doorstep_check(self) -> bool | None:
        """Look once for a parcel left by someone who didn't ring. Returns whether one is there."""
        self._doorstep_pending = False
        if self._busy or not self.camera or not self.watches.spend():
            return None
        self._doorstep_last = dt_util.utcnow()
        self._busy = True
        try:
            result = await self._look(
                "Sentient: anything left on the doorstep?",
                "This is the doorstep just after someone walked up to the house and left without "
                "ringing. Is a parcel, package, bag or post visibly left on the doorstep or by the "
                "door? If so, describe it in a few plain words (e.g. 'a small brown box').",
                {
                    "parcel_left": {"description": "True if a parcel or post is visibly left", "required": True,
                                    "selector": {"boolean": {}}},
                    "description": {"description": "What was left, a few plain words", "required": False,
                                    "selector": {"text": {}}},
                },
            )
        except HomeAssistantError as err:
            _LOGGER.warning("Sentient could not check the doorstep: %s", err)
            return None
        finally:
            self._busy = False
        there = bool(result.get("parcel_left"))
        was = self.parcel_waiting
        self.parcel_waiting = there
        if there and not was:
            what = (result.get("description") or t("visitors.a_parcel")).strip().rstrip(".")
            self.log.appendleft({
                "name": None,
                "time": dt_util.now().isoformat(),
                "kind": "delivery",
                "description": t("visitors.left_without_ringing", what=what),
                "parcel": True,
                "parcel_left": True,
                "trigger": "doorstep",
            })
            await self.speech.async_say(t("visitors.parcel_left"), "normal", None, source="visitors")
        self._save()
        return there

    # ----- looking ---------------------------------------------------------------
    async def _look(self, task: str, instructions: str, structure: dict, camera: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_name": task,
            "instructions": instructions + i18n.write_in(),
            "structure": structure,
            "attachments": [
                {"media_content_id": _media_id(camera or self.camera), "media_content_type": "image/jpeg"}
            ],
        }
        if ai_entity := ai_task_entity(self.hass, self.options):
            data["entity_id"] = ai_entity
        response = await self.hass.services.async_call(
            "ai_task", "generate_data", data, blocking=True, return_response=True
        )
        result = (response or {}).get("data")
        return result if isinstance(result, dict) else {}

    async def async_visit(self, trigger: str | None = None) -> dict[str, Any] | None:
        if self._busy:
            return None
        if not self.watches.spend():
            _LOGGER.debug("Visitor not described: daily AI limit reached")
            return None
        camera = self.camera_for(trigger)
        place = self.place_for(camera)
        self._busy = True
        face_task = None
        try:
            picture = await self.picture_for(trigger, camera)
            if self.faces is not None and self.faces.enabled:
                face_task = self.hass.async_create_task(self._faces_in(picture), eager_start=False)
            result = await self._look(
                "Sentient: who's at the door",
                "This is a camera picture at the door or gate, taken as someone rang or was seen there. "
                "Describe who is there in a few plain words (e.g. 'a delivery driver holding a small "
                "parcel', 'a man in a red coat', 'two children'). Don't guess names. Set kind to the best "
                "match (nobody if no one is visible), and parcel to true if they are carrying or have put "
                "down a parcel or package.",
                {
                    "kind": {"description": "Best match", "required": True,
                             "selector": {"select": {"options": KINDS}}},
                    "description": {"description": "A few plain words", "required": True,
                                    "selector": {"text": {}}},
                    "parcel": {"description": "Carrying or put down a parcel", "required": False,
                               "selector": {"boolean": {}}},
                },
                picture,
            )
            face = await face_task if face_task is not None else None
        except HomeAssistantError as err:
            _LOGGER.warning("Sentient could not look at the door: %s", err)
            if face_task is not None:
                face_task.cancel()
            return None
        finally:
            self._busy = False
        kind = result.get("kind") if result.get("kind") in KINDS else "unknown"
        # the camera's own face recognition first; Sentient's when the camera couldn't name anyone
        name = self.recognised_name(trigger) or (face or {}).get("name")
        household = self.is_household(name) or (not name and self.household_moving())
        visit = {
            "name": name,
            "time": dt_util.now().isoformat(),
            "kind": kind,
            "description": (result.get("description") or t("visitors.someone")).strip().rstrip("."),
            "parcel": bool(result.get("parcel")),
            "parcel_left": False,
            "trigger": trigger,
            "place": place,
            "household": household,
            "face_id": (face or {}).get("id"),  # so "that was Kata" can teach this face afterwards
        }
        self.log.appendleft(visit)
        self._save()
        if household:
            return visit  # logged, but nobody needs telling that they themselves came home
        if self.options.get(CONF_ANNOUNCE_VISITORS, DEFAULT_ANNOUNCE_VISITORS) and kind != "nobody":
            if place:
                line = (t("visitors.named_at_place", name=name, place=place) if name
                        else t("visitors.someone_at_place", description=visit["description"], place=place))
            else:
                line = (t("visitors.named_at_door", name=name) if name
                        else t("visitors.someone_at_door", description=visit["description"]))
            await self.speech.async_say(line, "normal", None, source="visitors")
        if (kind == "delivery" or visit["parcel"]) and self.options.get(CONF_PARCEL_ALERTS, DEFAULT_PARCEL_ALERTS):
            @callback
            def _recheck(_now: datetime) -> None:
                self.hass.async_create_task(self._async_parcel_check(visit, camera))

            async_call_later(self.hass, PARCEL_RECHECK, _recheck)
        return visit

    async def _async_parcel_check(self, visit: dict[str, Any], camera: str | None = None) -> None:
        if not self.watches.spend():
            return
        try:
            result = await self._look(
                "Sentient: parcel left?",
                "This is the doorstep or gate a few minutes after a delivery. Is a parcel, package or bag "
                "left there, on the doorstep or by the door or gate?",
                {"parcel_left": {"description": "True if a parcel is visibly left", "required": True,
                                 "selector": {"boolean": {}}}},
                camera,
            )
        except HomeAssistantError as err:
            _LOGGER.warning("Sentient could not check for a parcel: %s", err)
            return
        if result.get("parcel_left"):
            visit["parcel_left"] = True
            self.parcel_waiting = True
            self._save()
            await self.speech.async_say(t("visitors.parcel_left"), "normal", None, source="visitors")
