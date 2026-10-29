"""The language Sentient speaks in when it writes a sentence itself.

The house's language comes from Home Assistant (Settings → System → General → Language) and is
set once at start-up with ``set_language``. English stays the default, so nothing changes for an
English-speaking house. Text the AI writes is steered with ``write_in()``, which every AI task
prompt ends with.

Hungarian sentences put the device or room name first (with the right article) so they read
naturally whatever the name is.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

_LANGUAGE_NAMES = {"en": "British English", "hu": "Hungarian"}
_current = "en"


def set_language(code: str | None) -> None:
    """Pick the house language from Home Assistant's language code ('hu', 'en-GB', ...)."""
    global _current  # noqa: PLW0603 - one house, one language
    base = (code or "en").replace("_", "-").split("-")[0].lower()
    _current = base if base in _LANGUAGE_NAMES else "en"


def lang() -> str:
    return _current


def is_hu() -> bool:
    return _current == "hu"


def language_name() -> str:
    return _LANGUAGE_NAMES[_current]


def write_in() -> str:
    """The sentence every AI prompt ends with, so what it writes is in the house's language."""
    return f" Always write in {language_name()}."


# ----- Hungarian grammar helpers --------------------------------------------------
_VOWELS = "aáeéiíoóöőuúüű"


def article(name: str) -> str:
    """'a konyhai lámpa' / 'az előszobai lámpa' — Hungarian definite article plus the name."""
    name = (name or "").strip()
    if not name:
        return name
    lowered = name[0].lower() + name[1:]
    first = lowered[0]
    if first in _VOWELS:
        return f"az {lowered}"
    if first.isdigit():
        # 1 and 5 start with a vowel when read out (egy, öt)
        return f"az {lowered}" if first in "15" else f"a {lowered}"
    return f"a {lowered}"


def capital(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def join(names: list[str]) -> str:
    """'A, B and C' / 'A, B és C'."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    word = " és " if is_hu() else " and "
    return ", ".join(names[:-1]) + word + names[-1]


# ----- times and dates ---------------------------------------------------------------
_HU_DAYS = ["hétfő", "kedd", "szerda", "csütörtök", "péntek", "szombat", "vasárnap"]
_HU_MONTHS = ["január", "február", "március", "április", "május", "június", "július",
              "augusztus", "szeptember", "október", "november", "december"]


def clock(hour: int, minute: int) -> str:
    """'5:30pm' in English (voice models read it reliably), '17:30' in Hungarian."""
    if is_hu():
        return f"{hour}:{minute:02d}"
    h12 = hour % 12 or 12
    return f"{h12}{'' if minute == 0 else f':{minute:02d}'}{'am' if hour < 12 else 'pm'}"


def day_and_date(local: datetime) -> str:
    """'Friday 23 October' / 'október 23., péntek'."""
    if is_hu():
        return f"{_HU_MONTHS[local.month - 1]} {local.day}., {_HU_DAYS[local.weekday()]}"
    return f"{local:%A} {local.day} {local:%B}"


# ----- states, for sentences like "the bedroom temperature is now 24 °C" -------------
_HU_STATES = {
    "on": "bekapcsolva", "off": "kikapcsolva", "open": "nyitva", "opening": "nyílik", "closed": "zárva",
    "closing": "záródik", "locked": "bezárva", "unlocked": "kinyitva", "home": "otthon", "not_home": "távol",
    "idle": "tétlen", "playing": "lejátszás", "paused": "szüneteltetve", "standby": "készenlétben",
    "heating": "fűt", "cooling": "hűt", "docked": "dokkolva", "cleaning": "takarít", "returning": "visszatér",
    "unavailable": "nem elérhető", "unknown": "ismeretlen", "detected": "észlelve", "clear": "nincs észlelés",
}


def state_word(state: str) -> str:
    return _HU_STATES.get(state, state) if is_hu() else state


# ----- words that say something has finished (to catch an AI saying "done" too early) ---
FINISHED_WORDS = re.compile(
    r"\b(finished|done|ready|complete|completed|boiled|has stopped"
    r"|kész|elkészült|végzett|befejez\w*|lejárt|leállt|megállt|felforrt)\b",
    re.I,
)


# ----- sentences ------------------------------------------------------------------------
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # safety alerts
        "critical.smoke": "Smoke detected",
        "critical.carbon_monoxide": "Carbon monoxide detected",
        "critical.gas": "Gas detected",
        "critical.moisture": "Water leak detected",
        "critical.safety": "Safety alarm",
        "critical.tamper": "Tamper alarm",
        "critical.default": "Alarm",
        "critical.panel": "The {name_lower} has gone off!",
        "critical.named": "Alert: {name}!",
        "critical.in_area": "{base} in the {where_lower}!",
        "critical.at": "{base}: {where}!",
        "critical.cleared": "{name} has cleared",
        "critical.test": "This is a test of the safety alert. Nothing is wrong.",
        "critical.tested": "Safety alert tested",
        "critical.tested_from": "from the settings page",
        # left open
        "left_open.minutes": "{n} minutes",
        "left_open.hour": "{n} hour",
        "left_open.hours": "{n} hours",
        "left_open.open_for": "{name} has been open for {span}.",
        "left_open.last_left": "when the last person left",
        # night watch
        "night.noted": "{name} opened at {time}, not a usual time",
        "night.someone_up": "someone was up inside first ({inside}), so not alerting",
        "night.alert": "{name} opened at {time}. Nobody was moving inside and it doesn't normally open at this time.",
        "night.detail": "night watch",
        "night.overnight_alert": "{name} opened at {time} with nobody up (alerted)",
        "night.overnight_up": "{name} opened at {time} (someone was up)",
        "night.spoken": "Heads up: the {name_lower} has just opened, and nobody's been moving inside.",
        # visitors
        "visitors.parcel_left": "A parcel's been left at the front door.",
        "alert.heads_up": "Heads up: {text}.",
        "camsec.loitering": "Someone has been by the {place_lower} for a few minutes",
        "camsec.fence": "Someone is in the {place_lower}, but the gate and doors didn't open",
        "camsec.night": "Someone is by the {place_lower} and nobody's up",
        "alarm.default": "It's {time}. {name}.",
        "offer.default": "Shall I {name_lower}?",
        "follow.failed": "Sorry, the {what_lower} didn't respond, so it isn't {verb}.{why}",
        "follow.offline": " It looks offline. If it runs on batteries, they may be flat.",
        "follow.not_received": " It may not have received the command.",
        "follow.to_temperature": "set to {value} degrees",
        "follow.on": "on", "follow.off": "off", "follow.open": "open", "follow.closed": "closed",
        "follow.locked": "locked", "follow.unlocked": "unlocked",
        "visitors.named_at_door": "{name}'s at the door.",
        "visitors.someone_at_door": "Someone's at the door: {description}.",
        "visitors.named_at_place": "{place}: {name} is there.",
        "visitors.someone_at_place": "{place}: someone's there — {description}.",
        "visitors.someone": "someone at the door",
        "visitors.just_someone_at_door": "Someone's at the door.",
        "visitors.just_someone_at_place": "{place}: someone's there.",
        "visitors.a_parcel": "a parcel",
        "visitors.left_without_ringing": "someone left {what} without ringing",
        # welcome
        "welcome.plain": "Welcome back.",
        # learning
        "learning.every day": "every day",
        "learning.on weekdays": "on weekdays",
        "learning.at weekends": "at weekends",
        "learning.routine": "The {name} usually {verb} around {when} {days}.",
        "learning.link": "When the {cause} {cause_verb}{dark}, the {name} usually {verb} {soon}.",
        "learning.after_dark": " after dark",
        "learning.straight_after": "straight after",
        "learning.within_minute": "within a minute",
        "learning.within_minutes": "within a couple of minutes",
        "learning.offer": "I've noticed {said}. Want me to do that for you from now on?",
        "learning.offer_link": "I've noticed that {said}. Want me to do that for you from now on?",
        "learning.how_sure_days": "{pct}% of days",
        "learning.how_sure_times": "{pct}% of the time ({seen} times)",
        # watches and reminders
        "when.today": "{clock} today",
        "when.tomorrow": "{clock} tomorrow",
        "when.other": "{clock} on {date}",
        "watch.now": "{name} is now {state}.",
        "watch.heads_up": "Heads up: {what}.",
        # digest
        "digest.quiet": "A quiet day — nothing to report.",
        "digest.nothing": "Nothing worth mentioning.",
        "digest.last_hours": "last {hours} hours",
        # routines (checks said after a named routine)
        "routine.still_open_one": "{names} is still open",
        "routine.still_open_many": "{names} are still open",
        "routine.unlocked_one": "{names} is unlocked",
        "routine.unlocked_many": "{names} are unlocked",
        "routine.still_on_one": "{names} is still on",
        "routine.still_on_many": "{names} are still on",
        "routine.more": "{n} more",
        # security summary
        "security.open_one": "{names} is open",
        "security.open_many": "{names} are open",
        "security.unlocked_one": "{names} is unlocked",
        "security.unlocked_many": "{names} are unlocked",
        "security.all_shut": "Everything's shut",
        "security.and_locked": " and locked",
        "security.alarm_set": " The alarm is set ({mode}).",
        "security.alarm_off": " The alarm is off.",
        # phone notification titles
        "notify.ALERT": "ALERT",
        "notify.Night watch": "Night watch",
        "notify.Left open": "Left open",
        "notify.Announcement": "Announcement",
        "notify.Held back for quiet hours": "Held back for quiet hours",
        "notify.Nobody heard this": "Nobody heard this",
        "notify.Parcel left": "Parcel left",
        "notify.At the door": "At the door",
        "notify.Watch": "Watch",
        "notify.Daily digest": "Daily digest",
        "notify.This week": "This week",
        "notify.House health": "House health",
        # heating
        "heating.warming": "Warming {name} to {comfort}° for when people are up",
        "heating.usually_up": "usually up around {clock}",
        # energy
        "energy.used_times": "{name} has used {kwh} kWh today, about {times}× its usual",
        "energy.used_unusual": "{name} has used {kwh} kWh today, when it's normally next to nothing",
        # house health
        "health.offline": "{name} is offline",
        "health.battery": "{name} is at {level}%",
        "health.low": "{name} is low",
        "suggest.speaker": "{name} could be one of the house's speakers",
        "suggest.appliance": "{name} could say when it's finished, or seems stuck",
        "suggest.animals": "{name} could warn when it's too cold or too hot for the animals",
        "suggest.door_camera": "{name} could be the doorbell camera — who's there, parcels left",
        "suggest.ai_task": "{name} could do the camera looks and background thinking",
        "suggest.now_using": "Now using {name}",
        "suggest.from": "from a suggestion",
        "word.and": " and ",
        "rain.one": "Rain's on the way in the next hour, and the {things} is open.",
        "rain.many": "Rain's on the way in the next hour, and the {things} are open.",
        "pets.cat_was": "A cat was by the hutch at {time}.",
        "pets.cat_log": "night cat watch",
        "pets.cat_now_loud": "Heads up: there's a cat by the hutch.",
        "pets.cat_now": "There's a cat by the hutch.",
        "bins.not_out": "{what} tomorrow, and the bins aren't out yet.",
        "bins.log": "bins check",
        "learning.seen_times": " ({n} times in the last two weeks)",
        "learning.evidence": "{seen} of the last {total} {noun}",
        "learning.n_days": "days",
        "learning.n_weekdays": "weekdays",
        "learning.n_weekend_days": "weekend days",
        "drive.arrived_whose": "{name}'s car arrived on the drive",
        "drive.left_whose": "{name}'s car left the drive",
        "notify.show_camera": "Show camera",
        "notify.snooze": "Snooze 10 min",
        "notify.done": "Done",
        "wizard.name.title": "Name & personality",
        "wizard.name.blurb": "What the house is called and how it talks. Every voice and every remark uses this.",
        "wizard.speakers.title": "Speakers",
        "wizard.speakers.blurb": "Where it talks. Voice speakers are ticked for you; add Echos or other speakers if you like. The default is used when it can't tell where people are.",
        "wizard.rooms.title": "Rooms & movement",
        "wizard.rooms.blurb": "Motion sensors inside the house \u2014 how it knows which room people are in, and when the house has gone to bed. Ticked for you from the rooms your speakers are in.",
        "wizard.quiet.title": "Quiet hours",
        "wizard.quiet.blurb": "When to keep it down: nothing but urgent messages between these times.",
        "wizard.ai.title": "The AI",
        "wizard.ai.blurb": "The AI you already have in Home Assistant. Background jobs (cameras, briefings, the nightly think) use an AI task; voices use a conversation agent, with Home Assistant's own as the backup.",
        "wizard.cameras.title": "Doorbell & drive",
        "wizard.cameras.blurb": "Optional. The doorbell camera and what rings it, for visitors and parcels; a camera that sees the drive, for cars.",
        "wizard.extras.title": "Extras",
        "wizard.extras.blurb": "Optional. Appliances to follow (finished, or stuck), animals living outside, energy use, a morning briefing, and alerts on a tablet screen.",
        "animals.cold": "It's down to {deg} degrees in {where} — worth checking the bedding.",
        "animals.freezing": "It's down to {deg} degrees in {where} and freezing — check the water bottles and bedding.",
        "animals.hot": "It's up to {deg} degrees in {where} — they'll want shade and fresh water.",
        "animals.overnight": "Overnight: {text}",
        "appliance.stuck": "{name} seems to be stuck.",
        "appliance.slow": "{name} has been going {ran}; it usually takes about {usual}. It may be stuck.",
        "appliance.finished": "{name}'s finished.",
        "appliance.finished_log": "{name} finished",
        "appliance.ran": "ran {ran}",
        "appliance.usually": " (usually {usual})",
        "appliance.min": "{m} min",
        "appliance.h_m": "{h}h {m}m",
        "appliance.h": "{h}h",
        "autooff.learned": "{name} was switched straight back on — {item} now waits {after} min",
        "autooff.off": "{names} off",
        "autooff.no_movement": "no movement for {m} min",
        "autooff.on_for": "on for {m} min",
        "autooff.why": "auto-off ({item}): {why}",
        "word.or": " or ",
        "away.on": "Away lights on: replaying last week's evening so the house looks lived in",
        "away.off": "Away lights off — {why}",
        "away.home": "someone's home",
        "away.moving": "someone's moving about inside",
        "away.light": "it's light",
        "follow.earlier": "Earlier: ",
        "follow.while_away": "While you were away: ",
        "follow.thought": "A thought from overnight: ",
        "follow.and_thought": "And a thought from overnight: ",
        "follow.now": "{name} is {state}",
        "follow.done": "{about}: done — {now}.",
        "follow.not_yet": "{about}: not yet — {now}.",
        "follow.no_sign": "No sign of {what} — I've stopped keeping an eye out.",
        "follow.nothing_came": "Nothing came of watching for {what}.",
        "follow.log": "follow-up",
        "follow.log_done": " · done",
        "follow.log_not": " · not done",
        "follow.watch_ended": "watch ended",
        "foresight.frost": "Frost's forecast for tomorrow morning, so the car may need defrosting",
        "foresight.frost_animals": " and the water in {where} may freeze.",
        "foresight.early": "Early start tomorrow: {what} at {when}.",
        "foresight.something": "something",
        "foresight.log": "a look at tomorrow",
        "sleep.asleep": "House asleep",
        "sleep.up": "Up and about",
        "sleep.routine": "the {name} routine",
        "sleep.still": "nothing moving downstairs or up",
        "sleep.moving": "moving about in two rooms",
        "reflect.memories": "{n} new memories",
        "reflect.memory": "1 new memory",
        "reflect.wondering": "wondering: {q}",
        "reflect.lessons": "learned from {n} corrections",
        "reflect.lesson": "learned from 1 correction",
        "reflect.nothing": "Thought over the day — nothing to add.",
        "reflect.log": "overnight reflection",
        "curious.caller": "Someone's been coming to the door on {day}, around {when}, for a few weeks now. Do you know who that is?",
        "curious.asked": "asked out of curiosity",
        "day.0": "Mondays",
        "day.1": "Tuesdays",
        "day.2": "Wednesdays",
        "day.3": "Thursdays",
        "day.4": "Fridays",
        "day.5": "Saturdays",
        "day.6": "Sundays",
        "drive.arrived": "A car arrived on the drive",
        "drive.now": " — now {description}",
        "drive.left": "A car left the drive",
        "drive.still": " — {description} still there",
        "drive.clear": " — it's clear now",
        "drive.count": "{n} on the drive",
        "voice.not_answering": "The AI isn't answering right now, so I can only do simple things like lights and heating.",
        "voice.backup": "Voice answered by the backup agent",
        "voice.ai_failed": "the AI failed: {why}",
        "voiceact.HassTurnOn": "Turned on",
        "voiceact.HassTurnOff": "Turned off",
        "voiceact.HassLightSet": "Set",
        "voiceact.HassToggle": "Toggled",
        "voiceact.HassClimateSetTemperature": "Set the temperature of",
        "voiceact.HassSetPosition": "Moved",
        "voiceact.HassVacuumStart": "Started",
        "voiceact.HassVacuumReturnToBase": "Sent home",
        "voiceact.HassMediaPause": "Paused",
        "voiceact.HassMediaUnpause": "Resumed",
        "voiceact.HassSetVolume": "Set the volume of",
        "voiceact.HassMediaNext": "Skipped on",
        "voiceact.more": " and {n} more",
        "voiceact.by_voice": "by voice",
        "voiceact.didnt_work": "; didn't work: {names}",
        "voiceact.failed": "; failed: {err}",
        "voiceact.put_back": "Put back {names}",
        "voiceact.put_back_why": "the voice switched more than was asked for (\"{said}\")",
        "watchdog.stuck": "{name} has got stuck or not made it back {n} times this week",
        "watchdog.draining": "{name} is running down about {x}× faster than it used to",
        "watchdog.dropping": "{name} has dropped offline {n} times this week",
        "watchdog.fight": "{a} and {b} keep switching {name} back and forth ({n} times today)",
        "watchdog.frozen": "{name}'s picture hasn't changed for {hours} hours — it may be frozen",
        "heating.boosted": "{name} turned up — back to normal in {hours} hours",
        "heating.time_limit": "time limit",
        "camsec.nobody": "{text} — but nobody was in the picture ({time})",
        "plain.weather": "It's {weather} out.",
        "plain.reminders": "Reminders today: {items}.",
        "plain.open_one": "The {name} is open.",
        "plain.open_many": "Open now: {names}.",
        "plain.visitor": "1 visitor to the door today.",
        "plain.visitors": "{n} visitors to the door today.",
        "plain.nothing": "Nothing needs doing right now.",
        "welcome.plain_list": "Welcome back. While you were out: {items}.",
        "wizard.test": "Hello, I'm {persona}. This is how I'll sound in this room.",
        "wizard.done": "Set up with the wizard",
        "health.ai_failing": "The AI hasn't answered the last few times — plain words for now",
        "health.ai_capped": "Today's AI calls are used up — plain words until midnight",
        "health.speaker_failed": "Couldn't speak on {name}",
        "health.stuck_on": "{name} has said there's movement for {hours} hours",
        "health.stuck_quiet": "{name} hasn't seen anything for {days} days",
        # timeline
        "timeline.held": "held back: {reason}",
        "timeline.noticed_quiet": "noticed, decided it wasn't worth saying",
        "timeline.someone_at_door": "Someone at the door",
        "timeline.chime_on": "Chime on",
        "timeline.echos_on": "Echo speakers on",
        "timeline.echos_off": "Echo speakers off",
        "timeline.ai_back": "The AI is answering again",
        "timeline.speaker_back": "Speaking on {name} again",
        "timeline.sensor_back": "{name} is working again",
        "timeline.camera_back": "{name}'s picture is moving again",
        "timeline.fine_again": "{name} is fine again",
        "timeline.chime_off": "Chime off",
        "timeline.guest_on": "Guest mode on",
        "timeline.guest_off": "Guest mode off",
        "timeline.hushed_until": "Hushed until {time}",
        "timeline.talking_again": "Talking again",
        "timeline.back_online": "{name} is back to normal",
        "timeline.battery_ok": "{name}: battery fine again",
        # motion lights
        "motion.taken_over": "taken over by hand",
        "motion.quiet": "stayed off: quiet hours",
        "motion.on_late": "on at {pct}% (late night)",
        "motion.on": "on with motion",
        "motion.off": "off: no movement",
        # calendar tool
        "calendar.none": "There's no calendar to add events to",
        "calendar.not_found": "I can't find a calendar called '{name}'",
        "calendar.bad_time": "I couldn't understand the time '{value}'",
        "calendar.end_before_start": "The end is before the start",
    },
    "hu": {
        "critical.smoke": "Füstérzékelés",
        "critical.carbon_monoxide": "Szén-monoxid-érzékelés",
        "critical.gas": "Gázszivárgás",
        "critical.moisture": "Vízszivárgás",
        "critical.safety": "Biztonsági riasztás",
        "critical.tamper": "Szabotázsriasztás",
        "critical.default": "Riasztás",
        "critical.panel": "Megszólalt a riasztó: {name}!",
        "critical.named": "Riasztás: {name}!",
        "critical.in_area": "{base} itt: {where}!",
        "critical.at": "{base}: {where}!",
        "critical.cleared": "{name}: megszűnt a riasztás",
        "critical.test": "Ez a biztonsági riasztás próbája. Nincs semmi baj.",
        "critical.tested": "Biztonsági riasztás kipróbálva",
        "critical.tested_from": "a beállítások oldalról",
        "left_open.minutes": "{n} perce",
        "left_open.hour": "{n} órája",
        "left_open.hours": "{n} órája",
        "left_open.open_for": "{Name} már {span} nyitva van.",
        "left_open.last_left": "amikor az utolsó ember is elment",
        "night.noted": "{name} {time}-kor nyílt ki, szokatlan időben",
        "night.someone_up": "előtte már mozgott valaki bent ({inside}), ezért nincs riasztás",
        "night.alert": "{name} {time}-kor kinyílt. Bent senki nem mozgott, és ilyenkor általában nem nyílik ki.",
        "night.detail": "éjszakai őrség",
        "night.overnight_alert": "{name} {time}-kor nyílt ki, senki sem volt fent (riasztott)",
        "night.overnight_up": "{name} {time}-kor nyílt ki (valaki fent volt)",
        "night.spoken": "Figyelem: {name_art} most nyílt ki, és bent senki nem mozgott.",
        "visitors.parcel_left": "Csomagot hagytak a bejárati ajtónál.",
        "alert.heads_up": "Figyelem: {text}.",
        "camsec.loitering": "Pár perce valaki van itt: {place}",
        "camsec.fence": "Valaki van itt: {place}, pedig a kapu és az ajtók nem nyíltak ki",
        "camsec.night": "Valaki van itt: {place}, és senki sincs ébren",
        "alarm.default": "{time} van. {name}.",
        "offer.default": "Megcsináljam: {name_lower}?",
        "follow.failed": "Sajnos {what_art} nem reagált, így nem sikerült {verb}.{why}",
        "follow.offline": " Úgy tűnik, nem elérhető. Ha elemes, lehet, hogy lemerült.",
        "follow.not_received": " Lehet, hogy nem kapta meg a parancsot.",
        "follow.to_temperature": "{value} fokra állítani",
        "follow.on": "bekapcsolni", "follow.off": "kikapcsolni", "follow.open": "kinyitni", "follow.closed": "becsukni",
        "follow.locked": "bezárni", "follow.unlocked": "kinyitni",
        "visitors.named_at_door": "{name} van az ajtónál.",
        "visitors.someone_at_door": "Valaki az ajtónál van: {description}.",
        "visitors.named_at_place": "{place}: {name} érkezett.",
        "visitors.someone_at_place": "{place}: valaki jött – {description}.",
        "visitors.someone": "valaki az ajtónál",
        "visitors.just_someone_at_door": "Valaki az ajtónál van.",
        "visitors.just_someone_at_place": "{place}: valaki jött.",
        "visitors.a_parcel": "egy csomagot",
        "visitors.left_without_ringing": "valaki csengetés nélkül itt hagyott {what}",
        "welcome.plain": "Üdv itthon!",
        "learning.every day": "minden nap",
        "learning.on weekdays": "hétköznap",
        "learning.at weekends": "hétvégén",
        "learning.routine": "{Name_art} {days} általában {when} körül {verb}.",
        "learning.link": "Amikor {cause_art} {cause_verb}{dark}, {name_art} általában {soon} {verb}.",
        "learning.after_dark": " sötétedés után",
        "learning.straight_after": "rögtön utána",
        "learning.within_minute": "egy percen belül",
        "learning.within_minutes": "pár percen belül",
        "learning.offer": "Észrevettem, hogy {said}. Mostantól csináljam meg én?",
        "learning.offer_link": "Észrevettem, hogy {said}. Mostantól csináljam meg én?",
        "learning.how_sure_days": "a napok {pct}%-ában",
        "learning.how_sure_times": "az esetek {pct}%-ában ({seen} alkalom)",
        "when.today": "ma {clock}",
        "when.tomorrow": "holnap {clock}",
        "when.other": "{date} {clock}",
        "watch.now": "{name} most: {state}.",
        "watch.heads_up": "Figyelem: {what}.",
        "digest.quiet": "Csendes nap volt, nincs miről beszámolni.",
        "digest.nothing": "Nincs említésre méltó.",
        "digest.last_hours": "az elmúlt {hours} óra",
        "routine.still_open_one": "{names} még nyitva van",
        "routine.still_open_many": "{names} még nyitva vannak",
        "routine.unlocked_one": "{names} nincs bezárva",
        "routine.unlocked_many": "{names} nincsenek bezárva",
        "routine.still_on_one": "{names} még ég",
        "routine.still_on_many": "{names} még égnek",
        "routine.more": "még {n}",
        "security.open_one": "{names} nyitva van",
        "security.open_many": "{names} nyitva vannak",
        "security.unlocked_one": "{names} nincs bezárva",
        "security.unlocked_many": "{names} nincsenek bezárva",
        "security.all_shut": "Minden csukva van",
        "security.and_locked": " és bezárva",
        "security.alarm_set": " A riasztó élesítve ({mode}).",
        "security.alarm_off": " A riasztó ki van kapcsolva.",
        "notify.ALERT": "RIASZTÁS",
        "notify.Night watch": "Éjszakai őrség",
        "notify.Left open": "Nyitva maradt",
        "notify.Announcement": "Bejelentés",
        "notify.Held back for quiet hours": "Csendes órák miatt visszatartva",
        "notify.Nobody heard this": "Ezt senki nem hallotta",
        "notify.Parcel left": "Csomag érkezett",
        "notify.At the door": "Az ajtónál",
        "notify.Watch": "Figyelés",
        "notify.Daily digest": "Napi összefoglaló",
        "notify.This week": "Ezen a héten",
        "notify.House health": "A ház állapota",
        "heating.warming": "{name} felfűtése {comfort}°-ra, mire felkeltek",
        "heating.usually_up": "általában {clock} körül keltek",
        "energy.used_times": "{name} ma {kwh} kWh-t fogyasztott, a szokásos kb. {times}-szorosát",
        "energy.used_unusual": "{name} ma {kwh} kWh-t fogyasztott, pedig általában szinte semmit",
        "health.offline": "{name} nem elérhető",
        "health.battery": "{name}: akkumulátor {level}%",
        "health.low": "{name}: lemerülőben",
        "suggest.speaker": "{name} lehetne a ház egyik hangszórója",
        "suggest.appliance": "{name} szólhatna, ha végzett vagy elakadt",
        "suggest.animals": "{name} figyelmeztethetne, ha túl hideg vagy túl meleg van az állatoknak",
        "suggest.door_camera": "{name} lehetne a csengőkamera — ki jött, maradt-e csomag",
        "suggest.ai_task": "{name} végezhetné a kamerapillantásokat és a háttérgondolkodást",
        "suggest.now_using": "Mostantól használja: {name}",
        "suggest.from": "javaslat alapján",
        "word.and": " és ",
        "rain.one": "Egy órán belül eső jön, és nyitva van: {things}.",
        "rain.many": "Egy órán belül eső jön, és nyitva van: {things}.",
        "pets.cat_was": "{time}-kor egy macska járt az ólnál.",
        "pets.cat_log": "éjszakai macskafigyelés",
        "pets.cat_now_loud": "Figyelem: macska van az ólnál.",
        "pets.cat_now": "Macska van az ólnál.",
        "bins.not_out": "Holnap {what}, és még nincsenek kint a kukák.",
        "bins.log": "kuka-ellenőrzés",
        "learning.seen_times": " (az elmúlt két hétben {n}-szor)",
        "learning.evidence": "az utolsó {total} {noun} {seen}",
        "learning.n_days": "napból",
        "learning.n_weekdays": "hétköznapból",
        "learning.n_weekend_days": "hétvégi napból",
        "drive.arrived_whose": "{name} autója megérkezett a behajtóra",
        "drive.left_whose": "{name} autója elment a behajtóról",
        "notify.show_camera": "Kamera",
        "notify.snooze": "Szundi 10 perc",
        "notify.done": "Kész",
        "wizard.name.title": "Név és személyiség",
        "wizard.name.blurb": "Hogy hívják a házat, és hogyan beszél. Minden hang és megjegyzés ezt használja.",
        "wizard.speakers.title": "Hangszórók",
        "wizard.speakers.blurb": "Hol beszél. A hangasszisztens-hangszórók már be vannak jelölve; ha szeretnéd, adj hozzá Echókat vagy más hangszórókat. Az alapértelmezettet akkor használja, ha nem tudja, hol vannak az emberek.",
        "wizard.rooms.title": "Szobák és mozgás",
        "wizard.rooms.blurb": "Mozgásérzékelők a házban — ebből tudja, melyik szobában vannak az emberek, és mikor feküdt le a ház. A hangszórós szobák alapján előre bejelöltük.",
        "wizard.quiet.title": "Csendes órák",
        "wizard.quiet.blurb": "Mikor legyen csendben: ezek között az időpontok között csak sürgős üzenetek.",
        "wizard.ai.title": "Az AI",
        "wizard.ai.blurb": "A Home Assistantben már meglévő AI. A háttérfeladatok (kamerák, összefoglalók, éjszakai átgondolás) AI-feladatot használnak; a hangok beszélgetés-asszisztenst, tartaléknak a Home Assistant sajátjával.",
        "wizard.cameras.title": "Csengő és behajtó",
        "wizard.cameras.blurb": "Nem kötelező. A csengőkamera és ami megszólaltatja, a látogatókhoz és csomagokhoz; egy kamera, amely látja a behajtót, az autókhoz.",
        "wizard.extras.title": "Extrák",
        "wizard.extras.blurb": "Nem kötelező. Követett háztartási gépek (végzett vagy elakadt), kint élő állatok, energiafogyasztás, reggeli összefoglaló és riasztások tablet képernyőn.",
        "animals.cold": "{where} már csak {deg} fok van — érdemes megnézni az almot.",
        "animals.freezing": "{where} {deg} fok van, fagy — nézd meg az itatót és az almot.",
        "animals.hot": "{where} már {deg} fok van — árnyék és friss víz kell nekik.",
        "animals.overnight": "Az éjjel: {text}",
        "appliance.stuck": "{name} mintha elakadt volna.",
        "appliance.slow": "{name} már {ran} óta megy, általában kb. {usual}. Lehet, hogy elakadt.",
        "appliance.finished": "{name} végzett.",
        "appliance.finished_log": "{name} végzett",
        "appliance.ran": "{ran} ment",
        "appliance.usually": " (általában {usual})",
        "appliance.min": "{m} perc",
        "appliance.h_m": "{h} óra {m} perc",
        "appliance.h": "{h} óra",
        "autooff.learned": "{name} rögtön vissza lett kapcsolva — {item} mostantól {after} percet vár",
        "autooff.off": "{names} lekapcsolva",
        "autooff.no_movement": "{m} perce nincs mozgás",
        "autooff.on_for": "{m} perce ég",
        "autooff.why": "automatikus lekapcsolás ({item}): {why}",
        "word.or": " vagy ",
        "away.on": "Távolléti világítás be: a múlt heti estét játssza vissza, hogy lakottnak tűnjön a ház",
        "away.off": "Távolléti világítás ki — {why}",
        "away.home": "valaki hazaért",
        "away.moving": "valaki mozog bent",
        "away.light": "világos van",
        "follow.earlier": "Korábban: ",
        "follow.while_away": "Amíg nem voltál itt: ",
        "follow.thought": "Egy éjszakai gondolat: ",
        "follow.and_thought": "És egy éjszakai gondolat: ",
        "follow.now": "{name}: {state}",
        "follow.done": "{about}: kész — {now}.",
        "follow.not_yet": "{about}: még nem — {now}.",
        "follow.no_sign": "Nem láttam ezt: {what} — abbahagytam a figyelést.",
        "follow.nothing_came": "Nem lett eredménye a figyelésnek: {what}.",
        "follow.log": "visszatérés",
        "follow.log_done": " · kész",
        "follow.log_not": " · nincs kész",
        "follow.watch_ended": "figyelés vége",
        "foresight.frost": "Holnap reggelre fagyot mondanak, lehet, hogy jégteleníteni kell az autót",
        "foresight.frost_animals": ", és befagyhat a víz itt: {where}.",
        "foresight.early": "Korai kezdés holnap: {what}, {when}-kor.",
        "foresight.something": "valami",
        "foresight.log": "pillantás a holnapra",
        "sleep.asleep": "A ház alszik",
        "sleep.up": "Felébredtek",
        "sleep.routine": "a(z) {name} rutin",
        "sleep.still": "sehol nincs mozgás",
        "sleep.moving": "két szobában is mozgás van",
        "reflect.memories": "{n} új emlék",
        "reflect.memory": "1 új emlék",
        "reflect.wondering": "kíváncsi: {q}",
        "reflect.lessons": "{n} javításból tanult",
        "reflect.lesson": "1 javításból tanult",
        "reflect.nothing": "Átgondolta a napot — nincs mit hozzátenni.",
        "reflect.log": "éjszakai átgondolás",
        "curious.caller": "Pár hete {day} {when} körül valaki rendszeresen jön az ajtóhoz. Tudod, ki az?",
        "curious.asked": "kíváncsiságból kérdezte",
        "day.0": "hétfőnként",
        "day.1": "keddenként",
        "day.2": "szerdánként",
        "day.3": "csütörtökönként",
        "day.4": "péntekenként",
        "day.5": "szombatonként",
        "day.6": "vasárnaponként",
        "drive.arrived": "Autó érkezett a behajtóra",
        "drive.now": " — most: {description}",
        "drive.left": "Egy autó elment a behajtóról",
        "drive.still": " — {description} még ott van",
        "drive.clear": " — most üres",
        "drive.count": "{n} a behajtón",
        "voice.not_answering": "Az AI most nem válaszol, így csak egyszerű dolgokat tudok: lámpák, fűtés.",
        "voice.backup": "A tartalék asszisztens válaszolt",
        "voice.ai_failed": "az AI hibázott: {why}",
        "voiceact.HassTurnOn": "Bekapcsolva:",
        "voiceact.HassTurnOff": "Kikapcsolva:",
        "voiceact.HassLightSet": "Beállítva:",
        "voiceact.HassToggle": "Átkapcsolva:",
        "voiceact.HassClimateSetTemperature": "Hőmérséklet beállítva:",
        "voiceact.HassSetPosition": "Mozgatva:",
        "voiceact.HassVacuumStart": "Elindítva:",
        "voiceact.HassVacuumReturnToBase": "Hazaküldve:",
        "voiceact.HassMediaPause": "Szüneteltetve:",
        "voiceact.HassMediaUnpause": "Folytatva:",
        "voiceact.HassSetVolume": "Hangerő beállítva:",
        "voiceact.HassMediaNext": "Léptetve:",
        "voiceact.more": " és még {n}",
        "voiceact.by_voice": "hanggal",
        "voiceact.didnt_work": "; nem sikerült: {names}",
        "voiceact.failed": "; hiba: {err}",
        "voiceact.put_back": "Visszaállítva: {names}",
        "voiceact.put_back_why": "a hang többet kapcsolt, mint amit kértek („{said}”)",
        "watchdog.stuck": "{name} ezen a héten {n}-szor akadt el vagy nem ért vissza",
        "watchdog.draining": "{name} kb. {x}-szer gyorsabban merül, mint korábban",
        "watchdog.dropping": "{name} ezen a héten {n}-szor esett le a hálózatról",
        "watchdog.fight": "{a} és {b} folyton oda-vissza kapcsolja ezt: {name} (ma {n}-szor)",
        "watchdog.frozen": "{name} képe {hours} órája nem változott — lehet, hogy lefagyott",
        "heating.boosted": "{name} feljebb véve — {hours} óra múlva visszaáll",
        "heating.time_limit": "időkorlát",
        "camsec.nobody": "{text} — de senki sem volt a képen ({time})",
        "plain.weather": "Kint {weather}.",
        "plain.reminders": "Mai emlékeztetők: {items}.",
        "plain.open_one": "Nyitva van: {name}.",
        "plain.open_many": "Most nyitva: {names}.",
        "plain.visitor": "Ma egy látogató volt az ajtónál.",
        "plain.visitors": "Ma {n} látogató volt az ajtónál.",
        "plain.nothing": "Most nincs semmi teendő.",
        "welcome.plain_list": "Üdv itthon. Amíg nem voltál itthon: {items}.",
        "wizard.test": "Szia, {persona} vagyok. Így fogok szólni ebben a szobában.",
        "wizard.done": "Beállítva a varázslóval",
        "health.ai_failing": "Az AI az utóbbi néhány alkalommal nem válaszolt — addig egyszerű szavakkal",
        "health.ai_capped": "A mai AI-keret elfogyott — éjfélig egyszerű szavakkal",
        "health.speaker_failed": "Nem sikerült megszólalni ezen: {name}",
        "health.stuck_on": "{name} már {hours} órája mozgást jelez",
        "health.stuck_quiet": "{name} már {days} napja nem érzékelt semmit",
        "timeline.held": "visszatartva: {reason}",
        "timeline.noticed_quiet": "észrevette, de nem tartotta érdemesnek szólni",
        "timeline.someone_at_door": "Valaki az ajtónál",
        "timeline.chime_on": "Csengőhang be",
        "timeline.echos_on": "Echo hangszórók be",
        "timeline.echos_off": "Echo hangszórók ki",
        "timeline.ai_back": "Az AI újra válaszol",
        "timeline.speaker_back": "{name} újra megszólal",
        "timeline.sensor_back": "{name} újra működik",
        "timeline.camera_back": "{name} képe újra mozog",
        "timeline.fine_again": "{name} újra rendben",
        "timeline.chime_off": "Csengőhang ki",
        "timeline.guest_on": "Vendégmód be",
        "timeline.guest_off": "Vendégmód ki",
        "timeline.hushed_until": "Némítva {time}-ig",
        "timeline.talking_again": "Újra beszél",
        "timeline.back_online": "{name} újra rendben",
        "timeline.battery_ok": "{name}: akkumulátor újra rendben",
        "motion.taken_over": "kézzel átvéve",
        "motion.quiet": "kikapcsolva maradt: csendes órák",
        "motion.on_late": "bekapcsolt {pct}%-on (késő éjjel)",
        "motion.on": "mozgásra bekapcsolt",
        "motion.off": "kikapcsolt: nincs mozgás",
        "calendar.none": "Nincs olyan naptár, amibe eseményt lehetne felvenni",
        "calendar.not_found": "Nem találok '{name}' nevű naptárat",
        "calendar.bad_time": "Nem értem ezt az időpontot: '{value}'",
        "calendar.end_before_start": "A vége korábbi, mint a kezdete",
    },
}

_EN_VERBS = {"on": "goes on", "off": "goes off", "open": "opens", "closed": "closes"}
_HU_VERBS = {"on": "bekapcsol", "off": "kikapcsol", "open": "kinyílik", "closed": "bezáródik"}


def t(key: str, **kwargs: Any) -> str:
    """One sentence in the house's language.

    For every string argument (say ``name``) the template can also use ``{name_lower}``,
    ``{name_art}`` (Hungarian 'az előszobai lámpa'), ``{Name_art}`` and ``{Name}`` (capitalised).
    """
    template = STRINGS[_current].get(key) or STRINGS["en"][key]
    extra: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, str):
            extra[f"{k}_lower"] = v.lower()
            extra[f"{k}_art"] = article(v)
            extra[f"{capital(k)}_art"] = capital(article(v))
            extra[capital(k)] = capital(v)
    return template.format(**{**extra, **kwargs})


_HU_REASONS = {
    "quiet hours": "csendes órák", "hushed": "némítva", "duplicate": "ismétlés", "no speaker": "nincs hangszóró",
    "busy": "foglalt", "empty": "üres", "muted": "elhallgattatva", "hushed by": "csend egy automatizáció után",
}


def reason(code: str) -> str:
    """'quiet_hours:...' → 'quiet hours' / 'csendes órák' — why something was held back."""
    plain = str(code or "").split(":")[0].replace("_", " ")
    return _HU_REASONS.get(plain, plain) if is_hu() else plain


def title(text: str) -> str:
    """A phone notification title ('Parcel left' → 'Csomag érkezett'); unknown titles pass through."""
    return STRINGS[_current].get(f"notify.{text}", text)


def state_verb(state: str) -> str:
    """'goes on' / 'bekapcsol' — what a light, switch or cover does."""
    if is_hu():
        return _HU_VERBS.get(state, f"{state} állapotba vált")
    return _EN_VERBS.get(state, state)


def change_verb(domain: str, state: str, device_class: str | None, open_classes: set[str]) -> str:
    """The verb for something people change: a door opening, a speaker starting to play."""
    hu = is_hu()
    if domain == "binary_sensor":
        if device_class in open_classes or device_class is None:
            if hu:
                return "kinyílik" if state == "on" else "bezáródik"
            return "opens" if state == "on" else "closes"
        if hu:
            return "bekapcsol" if state == "on" else "kikapcsol"
        return "turns on" if state == "on" else "turns off"
    if domain == "media_player":
        if hu:
            return "elindítja a lejátszást" if state == "playing" else "kikapcsol"
        return "starts playing" if state == "playing" else "is turned off"
    if hu:
        return _HU_VERBS.get(state, f"{state} állapotba vált")
    return _EN_VERBS.get(state, f"changes to {state}")
