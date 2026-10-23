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
        "night.spoken": "Heads up: the {name_lower} has just opened, and nobody's been moving inside.",
        # visitors
        "visitors.parcel_left": "A parcel's been left at the front door.",
        "visitors.named_at_door": "{name}'s at the door.",
        "visitors.someone_at_door": "Someone's at the door: {description}.",
        "visitors.someone": "someone at the door",
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
        "night.spoken": "Figyelem: {name_art} most nyílt ki, és bent senki nem mozgott.",
        "visitors.parcel_left": "Csomagot hagytak a bejárati ajtónál.",
        "visitors.named_at_door": "{name} van az ajtónál.",
        "visitors.someone_at_door": "Valaki az ajtónál van: {description}.",
        "visitors.someone": "valaki az ajtónál",
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
