"""Sentient as an LLM API.

Any conversation agent (Google AI, OpenAI, Ollama, ...) can tick "Sentient"
under *Control Home Assistant*. It then sees the house's memories in its
prompt and gets tools to remember, recall, forget and speak through the house.
That gives every voice in the house one shared brain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util
from homeassistant.util.json import JsonObjectType

from .const import (
    personality_line,
    CONF_HOUSE_RULES,
    CONF_PERSONA,
    CONF_PROMPT_MEMORIES,
    DEFAULT_PERSONA,
    DEFAULT_PROMPT_MEMORIES,
    DOMAIN,
    LLM_API_ID,
    MEMORY_CATEGORIES,
    PRIORITIES,
)

from . import i18n
from .i18n import t
from .voice import VoiceIDError

if TYPE_CHECKING:
    from . import SentientRuntime


def _runtime(hass: HomeAssistant) -> SentientRuntime | None:
    entries = hass.config_entries.async_loaded_entries(DOMAIN)
    return entries[0].runtime_data if entries else None


class _Tool(llm.Tool):
    """Every Sentient tool says which integration provides it."""

    integration = DOMAIN


def _read_only() -> Any:
    """Tool annotations for look-up tools (newer Home Assistant only)."""
    if (annotations := getattr(llm, "ToolAnnotations", None)) is None:
        return None
    return annotations(read_only=True, destructive=False, idempotent=True, open_world=False)


class RememberTool(_Tool):
    name = "sentient_remember"
    description = (
        "Save a durable fact to the house's long-term memory so every voice and "
        "automation knows it later. Use for preferences, people, routines and facts "
        "the household tells you, or asks you to remember. One short sentence. "
        "Never store passwords, card or account numbers, or health details."
    )
    parameters = vol.Schema(
        {
            vol.Required("text"): cv.string,
            vol.Optional("category", default="fact"): vol.In(MEMORY_CATEGORIES),
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        memory = rt.memory.add(
            tool_input.tool_args["text"],
            tool_input.tool_args.get("category", "fact"),
            source=llm_context.platform,
        )
        return {"saved": memory.text, "id": memory.id}


class RecallTool(_Tool):
    annotations = _read_only()
    name = "sentient_recall"
    description = (
        "Search the house's long-term memory — by meaning, so describe what you're looking for in a few words. "
        "Use before answering questions about the household's people, preferences, routines, stories being told "
        "or anything you were told earlier."
    )
    parameters = vol.Schema({vol.Required("query"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        found = await rt.memory.async_search(tool_input.tool_args["query"])
        return {"memories": [{"id": m.id, "text": m.text, "category": m.category} for m in found]}


class ForgetTool(_Tool):
    name = "sentient_forget"
    description = (
        "Remove something from the house's long-term memory when asked to forget it. Pass its id (from "
        "sentient_recall) or its words; if it isn't clear which memory is meant, you get the closest few back to ask about."
    )
    parameters = vol.Schema({vol.Required("what"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        removed, closest = await rt.memory.async_forget(tool_input.tool_args["what"])
        if removed is None and closest:
            return {"forgotten": None, "not_sure_which": [{"id": m.id, "text": m.text} for m in closest],
                    "hint": "Ask which one, then call again with its id."}
        return {"forgotten": removed.text if removed else None}


class AnnounceTool(_Tool):
    name = "sentient_announce"
    description = (
        "Say something out loud through the house's speakers, e.g. 'tell the kitchen "
        "dinner is ready'. Leave 'speaker' empty to use the room where someone was last "
        "seen. Respects quiet hours unless priority is urgent."
    )
    parameters = vol.Schema(
        {
            vol.Required("message"): cv.string,
            vol.Optional("speaker"): cv.string,
            vol.Optional("priority", default="normal"): vol.In(PRIORITIES),
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        args = tool_input.tool_args
        targets = None
        if args.get("speaker"):
            found = rt.speech.find_speaker(args["speaker"])
            if found is None:
                return {"spoken": False, "reason": f"no speaker matches '{args['speaker']}'",
                        "speakers": rt.speech.describe_rooms()}
            targets = [found]
        result = await rt.speech.async_say(
            args["message"], args.get("priority", "normal"), targets, source="conversation"
        )
        return result.as_dict()


class WatchTool(_Tool):
    name = "sentient_watch"
    description = (
        "Keep an eye on something and speak up when it happens. Two kinds: "
        "(1) a device or sensor reaching a state or crossing a number, e.g. 'tell me when the "
        "washing machine finishes' (entity + to_state) or 'if the bedroom goes above 24' "
        "(entity + above); (2) a camera, e.g. 'watch the garden until 6 and tell me if the cats "
        "go near the rabbit pen' (entity = the camera or its area name, question = a yes/no "
        "question about the image, look_for = the object, e.g. cat, person, car). "
        "Use 'until' for a clock time (e.g. '18:00') or 'minutes'. For a camera, 'wake_on' "
        "can name a zone or object sensor (e.g. 'front garden person occupancy') so it only "
        "looks when something enters that zone. Confirm briefly using the local time "
        "returned; never read the id aloud. For a plain reminder at a time, use sentient_remind. "
        "If a named spot (see sentient_spot) is mentioned, only that part of the picture is checked."
    )
    parameters = vol.Schema(
        {
            vol.Required("what"): cv.string,
            vol.Required("entity"): cv.string,
            vol.Optional("to_state"): cv.string,
            vol.Optional("above"): vol.Coerce(float),
            vol.Optional("below"): vol.Coerce(float),
            vol.Optional("question"): cv.string,
            vol.Optional("look_for"): cv.string,
            vol.Optional("message"): cv.string,
            vol.Optional("until"): cv.string,
            vol.Optional("minutes"): vol.Coerce(float),
            vol.Optional("repeat"): cv.boolean,
            vol.Optional("priority"): vol.In(PRIORITIES),
            vol.Optional("wake_on"): cv.string,
            vol.Optional("spot"): cv.string,
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        args = dict(tool_input.tool_args)
        try:
            watch = rt.watches.add(args.pop("what"), args.pop("entity"), **args)
        except HomeAssistantError as err:
            return {"error": str(err)}
        return {"watching": watch.summary()}


class RemindTool(_Tool):
    name = "sentient_remind"
    description = (
        "Remind someone of something at a time, spoken in whichever room people are in, e.g. "
        "'remind me at 6 to put the oven on' (at='18:00') or 'in 20 minutes check the "
        "washing' (minutes=20). 'message' is the exact sentence to say then, written to the "
        "listener (e.g. 'Time to put the oven on.'). Confirm with the local time returned."
    )
    parameters = vol.Schema(
        {
            vol.Required("message"): cv.string,
            vol.Optional("at"): cv.string,
            vol.Optional("minutes"): vol.Coerce(float),
            vol.Optional("speaker"): cv.string,
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        args = tool_input.tool_args
        try:
            reminder = rt.watches.add_reminder(
                args["message"], until=args.get("at"), minutes=args.get("minutes"), speaker=args.get("speaker")
            )
        except HomeAssistantError as err:
            return {"error": str(err)}
        return {"reminder": reminder.summary()}


def _writable_calendars(hass: HomeAssistant) -> list[Any]:
    """Calendars that can take new events (local calendars, Google, CalDAV...)."""
    from homeassistant.components.calendar import CalendarEntityFeature

    return [
        st for st in hass.states.async_all("calendar")
        if int(st.attributes.get("supported_features") or 0) & CalendarEntityFeature.CREATE_EVENT
    ]


def _parse_when(value: str) -> tuple[Any, bool]:
    """'2026-10-24 18:00' / '2026-10-24T18:00:00+02:00' → local datetime; '2026-10-24' → date (all day)."""
    text = (value or "").strip()
    if len(text) == 10:  # a plain date: all day
        if (day := dt_util.parse_date(text)) is not None:
            return day, True
        raise ValueError(value)
    if len(text) > 10 and text[10] == " ":
        text = text[:10] + "T" + text[11:]
    when = dt_util.parse_datetime(text)
    if when is None:
        raise ValueError(value)
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt_util.get_default_time_zone())
    return dt_util.as_local(when), False


class CalendarEventTool(_Tool):
    name = "sentient_add_calendar_event"
    description = (
        "Put an event in one of the household's calendars, e.g. 'put the dentist on Tuesday at 3' or "
        "'add the school play on 12 March'. 'start' (and optional 'end') are local times as "
        "'YYYY-MM-DD HH:MM', or just 'YYYY-MM-DD' for an all-day event; work them out from today's "
        "date. Without an end it lasts an hour. Leave 'calendar' empty unless they name one; if there "
        "is more than one calendar you get the list back — ask which. Confirm with what's returned."
    )
    parameters = vol.Schema(
        {
            vol.Required("summary"): cv.string,
            vol.Required("start"): cv.string,
            vol.Optional("end"): cv.string,
            vol.Optional("calendar"): cv.string,
            vol.Optional("description"): cv.string,
            vol.Optional("location"): cv.string,
        }
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        from datetime import timedelta

        args = tool_input.tool_args
        calendars = _writable_calendars(hass)
        if not calendars:
            return {"error": t("calendar.none")}
        wanted = (args.get("calendar") or "").strip()
        if wanted:
            key = wanted.lower()
            target = next(
                (c for c in calendars if c.entity_id == wanted or (c.name or "").lower() == key), None
            ) or next((c for c in calendars if key in (c.name or "").lower() or key in c.entity_id), None)
            if target is None:
                return {"error": t("calendar.not_found", name=wanted),
                        "calendars": [c.name for c in calendars]}
        elif len(calendars) == 1:
            target = calendars[0]
        else:
            return {"which_calendar": [c.name for c in calendars]}

        try:
            start, all_day = _parse_when(args["start"])
        except ValueError:
            return {"error": t("calendar.bad_time", value=args["start"])}
        data: dict[str, Any] = {"entity_id": target.entity_id, "summary": args["summary"]}
        if args.get("description"):
            data["description"] = args["description"]
        if args.get("location"):
            data["location"] = args["location"]
        if all_day:
            end = start + timedelta(days=1)  # the end date is exclusive
            if args.get("end"):
                try:
                    last, last_all_day = _parse_when(args["end"])
                except ValueError:
                    return {"error": t("calendar.bad_time", value=args["end"])}
                end = (last if last_all_day else last.date()) + timedelta(days=1)
                if end <= start:
                    return {"error": t("calendar.end_before_start")}
            data["start_date"], data["end_date"] = start.isoformat(), end.isoformat()
        else:
            end = start + timedelta(hours=1)
            if args.get("end"):
                try:
                    end, end_all_day = _parse_when(args["end"])
                except ValueError:
                    return {"error": t("calendar.bad_time", value=args["end"])}
                if end_all_day:
                    return {"error": t("calendar.bad_time", value=args["end"])}
            if end <= start:
                return {"error": t("calendar.end_before_start")}
            data["start_date_time"] = start.isoformat()
            data["end_date_time"] = end.isoformat()
        try:
            await hass.services.async_call("calendar", "create_event", data, blocking=True,
                                           context=llm_context.context)
        except HomeAssistantError as err:
            return {"error": str(err)}
        return {
            "added": args["summary"],
            "calendar": target.name,
            "start": data.get("start_date_time") or data["start_date"],
            "end": data.get("end_date_time") or data["end_date"],
            "all_day": all_day,
        }


class LearnVoiceTool(_Tool):
    name = "sentient_learn_voice"
    description = (
        "Learn someone's voice, so the house knows who is talking from then on. If the person talking is the one "
        "to learn ('learn my voice, I'm Sanya'), set this_is_them true. If they ask you to learn someone else's "
        "voice ('learn Lili's voice'), set it false and ask that person to talk to you next. Either way the next "
        "few sentences heard are taken as that person's voice, so tell them to chat for a few turns."
    )
    parameters = vol.Schema({vol.Required("name"): cv.string, vol.Optional("this_is_them", default=True): cv.boolean})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        try:
            return await rt.voice.async_learn(tool_input.tool_args["name"], tool_input.tool_args.get("this_is_them", True))
        except VoiceIDError as err:
            return {"error": str(err)}


class VoicesTool(_Tool):
    annotations = _read_only()
    name = "sentient_voices"
    description = "Whose voices the house knows, and who it thinks is talking right now."

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        heard = rt.voice.current()
        try:
            people = await rt.voice.async_people()
        except VoiceIDError as err:
            return {"error": str(err)}
        return {"voices": people, "talking_now": (heard.name if heard and (heard.known or heard.enrolling) else None)}


class ForgetVoiceTool(_Tool):
    name = "sentient_forget_voice"
    description = "Forget someone's voice completely (they can be learned again later)."
    parameters = vol.Schema({vol.Required("name"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        try:
            return await rt.voice.async_forget(tool_input.tool_args["name"])
        except VoiceIDError as err:
            return {"error": str(err)}


class LearnFaceTool(_Tool):
    name = "sentient_learn_face"
    description = (
        "Learn someone's face, so the house recognises them at the door and gate (and doesn't announce "
        "the family). Two ways: (1) they are in front of a camera right now ('Kata is at the gate, learn her "
        "face') — set camera to the camera's name or area; (2) a recent visitor was them ('that was "
        "Sanya', 'the last visitor was Szofi') — leave camera empty; visit 0 is the latest visit with a "
        "face, 1 the one before. A few pictures in different light make it reliable."
    )
    parameters = vol.Schema({
        vol.Required("name"): cv.string,
        vol.Optional("camera"): cv.string,
        vol.Optional("visit", default=0): vol.All(vol.Coerce(int), vol.Range(min=0, max=20)),
    })

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        from .watches import resolve_entity

        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        args = tool_input.tool_args
        camera = None
        if args.get("camera"):
            camera = resolve_entity(hass, args["camera"], "camera")
            if camera is None or not camera.startswith("camera."):
                return {"error": f"I can't find a camera called '{args['camera']}'"}
        try:
            return await rt.visitors.async_learn_face(args["name"], camera, int(args.get("visit") or 0))
        except HomeAssistantError as err:
            return {"error": str(err)}


class FacesTool(_Tool):
    annotations = _read_only()
    name = "sentient_faces"
    description = "Whose faces the house has learned (to recognise people at the door and gate)."

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        try:
            return {"faces": await rt.faces.async_people()}
        except HomeAssistantError as err:
            return {"error": str(err)}


class ForgetFaceTool(_Tool):
    name = "sentient_forget_face"
    description = "Forget someone's face completely (it can be learned again later)."
    parameters = vol.Schema({vol.Required("name"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        try:
            return {"forgotten": await rt.faces.async_forget(tool_input.tool_args["name"])}
        except HomeAssistantError as err:
            return {"error": str(err)}


class ListWatchesTool(_Tool):
    annotations = _read_only()
    name = "sentient_watches"
    description = "List what the house is keeping an eye on and any reminders, with local times."
    parameters = vol.Schema({})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        return {"watches": [w.summary() for w in rt.watches.watches]}


class CancelWatchTool(_Tool):
    name = "sentient_cancel_watch"
    description = "Stop keeping an eye on something. 'what' can be words from the watch, e.g. 'the cats'."
    parameters = vol.Schema({vol.Required("what"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        removed = rt.watches.cancel(tool_input.tool_args["what"])
        return {"cancelled": removed.what if removed else None}


class LookTool(_Tool):
    name = "sentient_look"
    description = (
        "Look through a camera right now and answer a question about what it sees, e.g. "
        "'what's going on in the garden?', 'is there a parcel at the front door?', 'are the "
        "bins out?', 'is the car on the drive?'. 'camera' can be the camera's name or its area. If the "
        "question is about a named spot (the drive, the rabbit pen…), pass it as 'spot' — the right camera "
        "is used automatically. Answer the person in your own words from the result."
    )
    parameters = vol.Schema(
        {vol.Required("camera"): cv.string, vol.Required("question"): cv.string, vol.Optional("spot"): cv.string}
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        try:
            args = tool_input.tool_args
            return await rt.watches.async_look(args["camera"], args["question"], args.get("spot"))
        except HomeAssistantError as err:
            return {"error": str(err)}


class RoutinesTool(_Tool):
    annotations = _read_only()
    name = "sentient_routines"
    description = (
        "List the household routines learned from history (e.g. 'the hallway light usually goes "
        "on around 5:30pm on weekdays') and which ones the house now does by itself."
    )
    parameters = vol.Schema({})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        return {"routines": [r.summary() for r in rt.learner.all()]}


class AdoptRoutineTool(_Tool):
    name = "sentient_adopt_routine"
    description = (
        "Start (adopt=true) or stop (adopt=false) the house doing a learned routine by itself, "
        "e.g. after someone says yes to 'want me to do that for you?', or 'stop turning the "
        "hallway light on for me'. 'routine' is its id or words from it."
    )
    parameters = vol.Schema({vol.Required("routine"): cv.string, vol.Optional("adopt", default=True): cv.boolean})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        args = tool_input.tool_args
        routine = rt.learner.adopt(args["routine"], args.get("adopt", True))
        if routine is None:
            return {"error": "No learned routine matches that"}
        return {"routine": routine.describe(), "house_does_it": routine.adopted}


class ForgetRoutineTool(_Tool):
    name = "sentient_forget_routine"
    description = (
        "Delete a learned routine or habit for good when someone doesn't want it ('forget that the "
        "kitchen light goes on at six', 'that's not a routine, delete it'). The house stops doing it "
        "and never learns or offers it again. 'routine' is its id or words from it."
    )
    parameters = vol.Schema({vol.Required("routine"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        item = rt.learner.dismiss(tool_input.tool_args["routine"])
        if item is None:
            return {"error": "No learned routine matches that"}
        return {"deleted": item.describe()}


class HeatingTool(_Tool):
    name = "sentient_heating"
    description = (
        "What the house has learned about the heating: how fast each zone warms up, the temperature it's "
        "usually set to, when people usually get up, when it would need to start to be warm by then, and "
        "whether it warms up early by itself. Read-only."
    )
    parameters = vol.Schema({})
    annotations = _read_only()

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        return rt.heating.report() if rt is not None else {"error": "Sentient is not loaded"}


class NightWatchTool(_Tool):
    name = "sentient_night_watch"
    description = (
        "The night watch: which doors and gates it watches, recent openings at unusual times (and whether "
        "anyone was up inside first), and whether it says alerts out loud. For 'did anything happen in the "
        "night?' or 'was that gate opening anything to worry about?'. Read-only."
    )
    parameters = vol.Schema({})
    annotations = _read_only()

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        return {**rt.night_watch.describe(), "camera_security": rt.camsec.describe()}


class WeeklyTool(_Tool):
    name = "sentient_weekly_report"
    description = (
        "What the house learned and noticed this week: new habits, what it does for people now, what it's "
        "learned to keep quiet about, heating, night watch, new memories and how much it spoke. For "
        "'what have you learned this week?'. Summarise it in a few sentences."
    )
    parameters = vol.Schema({})
    annotations = _read_only()

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        from .weekly import build

        return build(rt)


class AcceptOfferTool(_Tool):
    name = "sentient_accept_offer"
    description = (
        "The answer to one of the house's offers ('fancy a brew?', 'shall I put the heating on?'): "
        "accept=true does it, accept=false is a no (and it'll ask less). 'offer' is its id or name; "
        "'note' is any detail they gave."
    )
    parameters = vol.Schema({vol.Required("offer"): cv.string, vol.Optional("accept", default=True): cv.boolean,
                             vol.Optional("note"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        a = tool_input.tool_args
        return await rt.offers.async_accept(a["offer"], a.get("accept", True), a.get("note"))


class AlarmTool(_Tool):
    name = "sentient_alarm"
    description = (
        "Alarms and appointments. action 'set': name, time (HH:MM, 24-hour), and either days "
        "(list of mon..sun, or 'weekdays'/'weekends'/'every day') for a repeating alarm, or date "
        "(YYYY-MM-DD) for a one-off on another day (neither = next time it comes round); optional "
        "description and routine (the name of one of the household's named routines to run when it "
        "goes off). action 'list' shows them and when each next goes off. action 'cancel' with name "
        "deletes one; 'off'/'on' with name pauses or resumes it."
    )
    parameters = vol.Schema({
        vol.Required("action"): vol.In(["set", "list", "cancel", "off", "on"]),
        vol.Optional("name"): cv.string, vol.Optional("time"): cv.string,
        vol.Optional("days"): vol.Any(cv.string, [cv.string]), vol.Optional("date"): cv.string,
        vol.Optional("description"): cv.string, vol.Optional("routine"): cv.string,
    })

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        a = tool_input.tool_args
        action = a["action"]
        if action == "list":
            return {"alarms": rt.alarms.listing()}
        if not a.get("name"):
            return {"error": "Which alarm?"}
        if action in ("cancel", "off", "on"):
            item = rt.lists["alarms"].find(a["name"])
            if item is None:
                return {"error": f"No alarm called {a['name']}", "alarms": [x["name"] for x in rt.alarms.listing()]}
            if action == "cancel":
                rt.lists["alarms"].remove(item["id"])
                return {"cancelled": item["name"]}
            rt.lists["alarms"].update(item["id"], enabled=action == "on")
            return {"alarm": item["name"], "on": action == "on"}
        if not a.get("time"):
            return {"error": "What time?"}
        from .kinds import DAYS

        days = a.get("days") or []
        if isinstance(days, str):
            low = days.lower()
            days = (DAYS if "every" in low or "daily" in low else DAYS[:5] if "weekday" in low
                    else DAYS[5:] if "weekend" in low else [d for d in DAYS if d in low])
        if a.get("routine") and rt.shortcuts.find(a["routine"]) is None:
            return {"error": f"There's no routine called {a['routine']}", "routines": [r.name for r in rt.shortcuts.items.values()]}
        item = rt.alarms.set(a["name"], a["time"], days, a.get("date"), a.get("description") or "", a.get("routine"))
        nxt = next((x for x in rt.alarms.listing() if x["id"] == item["id"]), {})
        return {"set": item["name"], "time": str(item["time"])[:5], "days": item["days"], "next": nxt.get("next")}


class PartyTool(_Tool):
    name = "sentient_party"
    description = (
        "Light shows: lights put on a colour show, then go back exactly as they were. Use for 'disco "
        "mode', 'party lights', 'light show', 'chill lights', 'Halloween lights', 'Christmas lights' and "
        "'stop the disco'. action 'start' (optional style: disco, party, rainbow, chill, halloween or "
        "christmas; minutes; area = a room's lights, or 'everywhere' for every light in the house), 'stop', "
        "or 'status'. Starting again while it's on changes the style or room. If it says it's following the "
        "sound, the lights react to the music in the room."
    )
    parameters = vol.Schema({
        vol.Required("action"): vol.In(["start", "stop", "status"]),
        vol.Optional("style"): cv.string, vol.Optional("minutes"): vol.Coerce(float), vol.Optional("area"): cv.string,
    })

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        a = tool_input.tool_args
        if a["action"] == "status":
            return rt.party.describe()
        if a["action"] == "stop":
            return await rt.party.async_end()
        style = (a.get("style") or "").strip().lower() or None
        lights = rt.party.lights_for(a.get("area"), llm_context.device_id) if a.get("area") else None
        if a.get("area") and not lights:
            return {"error": f"No lights found in {a['area']}"}
        if lights is None and not rt.party.lights_for() and llm_context.device_id:
            lights = rt.party.lights_for(None, llm_context.device_id)
        return await rt.party.async_party(style, a.get("minutes"), lights)


class BriefingTool(_Tool):
    name = "sentient_briefing"
    description = (
        "Get what matters right now for a briefing or 'what's going on?': reminders due today, "
        "what the house is keeping an eye on, routines coming up, windows or doors open, the "
        "weather and what was said recently. Turn it into a few natural spoken sentences; "
        "skip anything empty."
    )
    parameters = vol.Schema({})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        from .upkeep import gather_briefing

        return await gather_briefing(hass, rt)


class VisitorsTool(_Tool):
    annotations = _read_only()
    name = "sentient_visitors"
    description = (
        "Who came to the door recently (from the doorbell camera), e.g. 'did anyone come round?', "
        "'has a parcel come?'. Gives local times and plain descriptions, never names."
    )
    parameters = vol.Schema({})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        return {"visitors": rt.visitors.summary(10)}


class SpotTool(_Tool):
    name = "sentient_spot"
    description = (
        "Teach the house a named area inside a camera picture, e.g. 'on the garden camera, the "
        "rabbit pen is a spot' — it finds it in the current picture and remembers where it is. "
        "Later watches and looks that mention it only check that part of the picture. Use "
        "forget=true to remove one, or no name to list the spots."
    )
    parameters = vol.Schema(
        {vol.Optional("camera"): cv.string, vol.Optional("name"): cv.string, vol.Optional("forget"): cv.boolean}
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        args = tool_input.tool_args
        if not args.get("name"):
            return {"spots": rt.spots.listing()}
        if args.get("forget"):
            return {"forgotten": rt.spots.forget(args["name"])}
        from .watches import resolve_entity

        camera = resolve_entity(hass, args.get("camera") or "", "camera")
        if camera is None or not camera.startswith("camera."):
            return {"error": "Which camera is it on?"}
        rt.watches.spend()
        try:
            spot = await rt.spots.async_learn(camera, args["name"])
        except HomeAssistantError as err:
            return {"error": str(err)}
        return {"learned": spot["name"], "camera": hass.states.get(camera).name}


class MotionLightsTool(_Tool):
    name = "sentient_motion_lights"
    description = (
        "Lights that come on with movement after dark. With no light, lists them. With a light and "
        "on=false/true, stops or restarts that light following motion, e.g. 'stop the landing light "
        "coming on with motion'."
    )
    parameters = vol.Schema({vol.Optional("light"): cv.string, vol.Optional("on"): cv.boolean})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        args = tool_input.tool_args
        ml = rt.motion_lights
        if not args.get("light") or "on" not in args:
            return {"motion_lights": ml.describe()}
        rule = ml.find(args["light"])
        if rule is None:
            return {"error": "No motion light matches that", "motion_lights": [d["light"] for d in ml.describe()]}
        ml.set_enabled(rule, bool(args["on"]))
        st = hass.states.get(rule.light)
        return {"light": st.name if st else rule.light, "follows_motion": rule.enabled}


class TimelineTool(_Tool):
    annotations = _read_only()
    name = "sentient_timeline"
    description = (
        "What the house did and noticed recently, newest first: what it said or held back, doorbell "
        "visitors, watches and reminders going off, motion lights, devices going offline. Use for "
        "'what happened while I was out?', 'did anything happen overnight?'."
    )
    parameters = vol.Schema({vol.Optional("hours", default=12): vol.Coerce(float)})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        hours = max(0.5, min(72.0, float(tool_input.tool_args.get("hours", 12))))
        items = rt.timeline.since(hours, ["said", "visitor", "watch", "light", "health"], 40)
        from .watches import local_when

        return {
            "hours": hours,
            "events": [
                {"when": local_when(dt_util.parse_datetime(i["time"])), "what": i["text"], **({"detail": i["detail"]} if i.get("detail") else {})}
                for i in items
            ],
        }


class QuietTool(_Tool):
    name = "sentient_quiet"
    description = (
        "Hush the house ('be quiet for an hour': minutes=60, 'you can talk again': minutes=0) — it then "
        "says nothing that isn't urgent. guests=true/false turns guest mode on or off: no remarks or "
        "routine offers, and personal household details stay private."
    )
    parameters = vol.Schema({vol.Optional("minutes"): vol.Coerce(float), vol.Optional("guests"): cv.boolean})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        from datetime import timedelta

        from .switch import hush, set_guest_mode
        from .watches import local_when

        entries = hass.config_entries.async_loaded_entries(DOMAIN)
        if not entries:
            return {"error": "Sentient is not loaded"}
        entry, args = entries[0], tool_input.tool_args
        out: dict[str, Any] = {}
        if "minutes" in args:
            minutes = max(0.0, min(720.0, float(args["minutes"])))
            until = dt_util.utcnow() + timedelta(minutes=minutes) if minutes else None
            hush(hass, entry, until)
            out["hushed_until"] = local_when(until) if until else None
        if "guests" in args:
            set_guest_mode(hass, entry, bool(args["guests"]))
            out["guest_mode"] = bool(args["guests"])
        return out or {"hushed": entry.runtime_data.speech.hushed(), "guest_mode": entry.runtime_data.speech.guest_mode}


class ShushTool(_Tool):
    name = "sentient_shush"
    description = (
        "Stop the house mentioning something: 'stop telling me about the washing machine' (what='washing "
        "machine' or its entity id), optionally for some hours ('for tonight' ≈ 12). undo=true lets it talk "
        "about it again (no 'what' clears every shush). No arguments lists what's shushed. Urgent and "
        "safety alerts always get through."
    )
    parameters = vol.Schema(
        {vol.Optional("what"): cv.string, vol.Optional("hours"): vol.Coerce(float), vol.Optional("undo"): cv.boolean}
    )

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        args = tool_input.tool_args
        if args.get("undo"):
            removed = rt.mutes.remove(args.get("what"))
            rt.timeline.add("mode", f"Unshushed: {args.get('what') or 'everything'}")
            return {"unshushed": removed}
        if not args.get("what"):
            return {"shushed": rt.mutes.listing(), "learned_from_feedback": rt.feedback.listing()}
        from .watches import resolve_entity

        target = resolve_entity(hass, args["what"]) or args["what"]
        mute = rt.mutes.add(target, args.get("hours"))
        name = hass.states.get(target).name if hass.states.get(target) else target
        rt.timeline.add("mode", f"Shushed: {name}", f"for {args['hours']:g} hours" if args.get("hours") else "until unshushed")
        return {"shushed": name, "until": mute["until"]}


class RunRoutineTool(_Tool):
    name = "sentient_run_routine"
    description = (
        "Run one of the household's named routines (e.g. 'goodnight', 'good morning', 'leaving'). "
        "Returns what to tell the person — say it in your reply (the house doesn't announce it)."
    )
    parameters = vol.Schema({vol.Required("name"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        if rt is None:
            return {"error": "Sentient is not loaded"}
        result = await rt.shortcuts.async_run(tool_input.tool_args["name"], llm_context.context)
        if not result.get("error"):
            rt.timeline.add("mode", f"Ran {result['routine']} (by voice)", result.get("said") or None)
        return result


class EnergyTool(_Tool):
    annotations = _read_only()
    name = "sentient_energy"
    description = (
        "Electricity used today by the devices on the Energy dashboard, compared with the past week at "
        "the same time of day: biggest users and anything using far more than usual. For 'what's using "
        "all the electricity?', 'has the heater been on a lot today?'."
    )
    parameters = vol.Schema({})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        from .energy import async_report, describe

        report = await async_report(hass)
        if not report.get("enabled", True):
            return {"error": report["note"]}
        return {
            "cost_today_gbp": report.get("cost_today"),
            "up_to": report.get("up_to"),
            "total_kwh_today": report.get("total_kwh"),
            "usual_by_now": report.get("usual_kwh"),
            "top": report.get("devices", [])[:6],
            "unusual": describe(report),
        }


class SecurityTool(_Tool):
    annotations = _read_only()
    name = "sentient_security"
    description = (
        "Is the house secure? Lists doors, windows, gates and garage doors that are open, locks that are "
        "unlocked, and the alarm's state. Read-only — it never locks, unlocks, arms or disarms anything. "
        "For 'is everything locked up?', 'have I left a window open?'."
    )
    parameters = vol.Schema({})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        from .security import check

        return check(hass)


class ScenesTool(_Tool):
    name = "sentient_scene"
    description = (
        "Set the mood: with no 'name', lists the home's scenes and named routines with what's in them, so "
        "you can pick the best one for a request like 'something cosy for movie night'. With 'name', "
        "turns that scene on (or runs that routine)."
    )
    parameters = vol.Schema({vol.Optional("name"): cv.string})

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        rt = _runtime(hass)
        scenes = [st for st in hass.states.async_all("scene")]
        name = (tool_input.tool_args.get("name") or "").strip()
        if not name:
            listing = []
            for st in scenes:
                members = st.attributes.get("entity_id") or []
                listing.append({
                    "name": st.name,
                    "contains": [hass.states.get(e).name if hass.states.get(e) else e for e in members][:8],
                })
            routines = [s.name for s in rt.shortcuts.items.values()] if rt else []
            return {"scenes": listing, "routines": routines}
        low = name.lower()
        match = next((st for st in scenes if st.entity_id == name or st.name.lower() == low), None) or next(
            (st for st in scenes if low in st.name.lower()), None
        )
        if match:
            await hass.services.async_call("scene", "turn_on", {"entity_id": match.entity_id}, blocking=True,
                                           context=llm_context.context)
            return {"scene_on": match.name}
        if rt and rt.shortcuts.find(name):
            return await rt.shortcuts.async_run(name, llm_context.context)
        return {"error": f"No scene or routine called {name}"}


class SentientAPI(llm.API):
    """Shared brain exposed to conversation agents."""

    def __init__(self, hass: HomeAssistant) -> None:
        super().__init__(hass=hass, id=LLM_API_ID, name="Sentient")

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        return llm.APIInstance(
            api=self,
            api_prompt=self._prompt(llm_context),
            llm_context=llm_context,
            tools=[RememberTool(), RecallTool(), ForgetTool(), AnnounceTool(),
                   WatchTool(), RemindTool(), ListWatchesTool(), CancelWatchTool(), LookTool(),
                   RoutinesTool(), AdoptRoutineTool(), ForgetRoutineTool(), AcceptOfferTool(), AlarmTool(), PartyTool(), HeatingTool(), NightWatchTool(), WeeklyTool(), BriefingTool(), VisitorsTool(), SpotTool(), MotionLightsTool(), TimelineTool(), QuietTool(), ShushTool(), RunRoutineTool(), EnergyTool(), SecurityTool(), ScenesTool(), CalendarEventTool()]
                  + ([LearnVoiceTool(), VoicesTool(), ForgetVoiceTool(), LearnFaceTool(), FacesTool(), ForgetFaceTool()]
                     if self._voice_on() else []),
        )

    def _voice_on(self) -> bool:
        rt = _runtime(self.hass)
        return bool(rt and rt.voice.enabled)

    @staticmethod
    def _speaker_lines(rt: SentientRuntime) -> list[str]:
        """Who is talking, from their voice (only when voice recognition is set up)."""
        voice = rt.voice
        if not voice.enabled:
            return []
        heard = voice.current()
        enrol = voice.enrolling()
        if heard and heard.enrolling and enrol:
            out = [f"You are learning {heard.name}'s voice: {enrol['left']} more sentence(s) needed. Just chat "
                   f"with {heard.name} in short, friendly turns."]
        elif heard and heard.enrolling:
            out = [f"You have just finished learning {heard.name}'s voice: tell them you'll know it's them now."]
        elif heard and heard.known and heard.name:
            out = [f"The person talking to you is {heard.name} (recognised by their voice). Talk to them as "
                   f"{heard.name}; use sentient_recall with their name for what you know about them (age, likes, "
                   "stories you're telling them)."]
        elif heard and heard.unsure and heard.name:
            out = [f"The person talking might be {heard.name}, but the voice wasn't clear enough to be sure: "
                   "don't share or act on anything personal on that guess."]
        else:
            out = ["You couldn't tell from the voice who is talking."]
        out.append(
            "When you tell someone a story, or play a game they'll want to carry on later, save where it got to with "
            "sentient_remember (category note), saying who it's for, e.g. \"Lili's dragon story: the dragon found "
            "the lost key, next she flies home\". To carry on, recall it by their name. To learn a voice, use "
            "sentient_learn_voice."
        )
        return out

    def _prompt(self, llm_context: llm.LLMContext) -> str:
        rt = _runtime(self.hass)
        if rt is None:
            return "Sentient (the house's long-term brain) is not loaded right now."
        opts: dict[str, Any] = rt.options
        persona = opts.get(CONF_PERSONA) or DEFAULT_PERSONA
        lines = [
            f"You are {persona}, the brain of this home. The same memory is shared by "
            "every voice and automation in the house.",
            personality_line(opts),
            f"Local time: {dt_util.now().strftime('%A %d %B %Y, %H:%M')}.",
        ]
        if rules := (opts.get(CONF_HOUSE_RULES) or "").strip():
            lines.append(f"House rules you must always follow:\n{rules}")
        room = self._room_of_device(llm_context.device_id)
        if room:
            lines.append(f"The person talking to you is in the {room}.")
        lines.extend(self._speaker_lines(rt))
        if rt.speech.guest_mode:
            lines.append(
                "Guests are visiting: don't bring up personal household details, routines or memories "
                "unless someone explicitly asks."
            )
        if rt.speech.hushed():
            lines.append("The house has been asked to stay quiet: keep replies short and don't announce anything.")
        elif rt.speech.quiet_now():
            lines.append("It is quiet hours: keep replies short and don't announce elsewhere unless urgent.")
        memories = rt.memory.recent(int(opts.get(CONF_PROMPT_MEMORIES, DEFAULT_PROMPT_MEMORIES)))
        if memories:
            lines.append("Things you know about this household:")
            lines.extend(f"- {m.text}" for m in memories)
            lines.append("Use sentient_recall for anything not listed here.")
        if spots := rt.spots.describe():
            lines.append(
                "Named camera spots — for questions about these, call sentient_look with the spot; it picks "
                "the camera that can see it: " + "; ".join(spots[:15]) + "."
            )
        offline = [i["name"] for i in rt.health.issues if i.get("kind") == "offline"]
        if offline:
            lines.append(
                "Offline right now (commands to these won't work — say so instead of claiming it's done): "
                + ", ".join(offline[:12]) + "."
            )
        lines.append(
            "After controlling a device, only say it's done if the tool result says it worked; if a device is "
            "unavailable or the action failed, say so plainly."
        )
        routines = sorted(rt.learner.all(), key=lambda r: (r.adopted, r.support), reverse=True)[:10]
        if routines:
            lines.append("Household routines and habits learned from history:")
            lines.extend(
                f"- {r.describe()}{' (the house does this now)' if r.adopted else ''}" for r in routines
            )
        if rt.shortcuts.items:
            lines.append(
                "Named routines you can run with sentient_run_routine: "
                + ", ".join(s.name for s in rt.shortcuts.items.values()) + "."
            )
        if problems := rt.health.describe():
            lines.append("House health problems right now: " + "; ".join(problems[:6]) + ".")
        if visits := rt.visitors.today():
            lines.append(f"Visitors to the door today: {len(visits)}. Use sentient_visitors for details.")
        watches = rt.watches.watches
        if watches:
            lines.append("You are currently keeping an eye on (and reminders due):")
            lines.extend(
                f"- {'reminder: ' if w.kind == 'reminder' else ''}{w.what} "
                f"({'at' if w.kind == 'reminder' else 'until'} {w.summary()['until']})"
                for w in watches[:10]
            )
        lines.append(
            "When someone asks you to keep an eye on something, or to tell them when something "
            "happens, use sentient_watch; to be reminded at a time, use sentient_remind; to put "
            "something in the calendar, use sentient_add_calendar_event. Never just promise to remember."
        )
        if i18n.lang() != "en":
            lines.append(f"Always speak and answer in {i18n.language_name()}.")
        lines.append(
            "When someone tells you something worth knowing later, or asks you to "
            "remember it, save it with sentient_remember. Never use the words "
            + ", ".join(f'"{w}"' for w in rt.speech.banned)
            + " when addressing people."
        )
        return "\n".join(lines)

    def _room_of_device(self, device_id: str | None) -> str | None:
        if not device_id:
            return None
        device = dr.async_get(self.hass).async_get(device_id)
        if device is None or not device.area_id:
            return None
        area = ar.async_get(self.hass).async_get_area(device.area_id)
        return area.name if area else None
