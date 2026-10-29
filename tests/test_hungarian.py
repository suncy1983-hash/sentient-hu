"""A Hungarian house: what Sentient says, how memory search copes, calendar events, admin-only calls."""

from datetime import datetime

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_mock_service

from custom_components.sentient import i18n
from custom_components.sentient.const import DOMAIN


@pytest.fixture(autouse=True)
def _back_to_english():
    yield
    i18n.set_language("en")


async def _setup(hass: HomeAssistant, language: str = "hu") -> MockConfigEntry:
    hass.config.language = language
    await hass.config.async_set_time_zone("Europe/Budapest")
    entry = MockConfigEntry(
        domain=DOMAIN, title="Nova", unique_id=DOMAIN,
        data={"persona": "Nova", "speakers": []}, options={},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def _ctx() -> llm.LLMContext:
    return llm.LLMContext(platform="test", context=None, language="hu", assistant="conversation", device_id=None)


# ----- language helpers -----------------------------------------------------------------
def test_language_comes_from_home_assistant() -> None:
    i18n.set_language("hu")
    assert i18n.lang() == "hu" and i18n.language_name() == "Hungarian"
    i18n.set_language("en-GB")
    assert i18n.lang() == "en"
    i18n.set_language("xx")  # anything we don't speak falls back to English
    assert i18n.lang() == "en"


def test_hungarian_article() -> None:
    assert i18n.article("Előszoba lámpa") == "az előszoba lámpa"
    assert i18n.article("Konyha lámpa") == "a konyha lámpa"
    assert i18n.article("1-es zóna") == "az 1-es zóna"
    assert i18n.article("2-es zóna") == "a 2-es zóna"


def test_spoken_sentences_in_hungarian() -> None:
    i18n.set_language("hu")
    assert i18n.t("left_open.open_for", name="Teraszajtó", span=i18n.t("left_open.minutes", n=10)) == (
        "Teraszajtó már 10 perce nyitva van."
    )
    assert i18n.t("night.spoken", name="Garázskapu") == "Figyelem: a garázskapu most nyílt ki, és bent senki nem mozgott."
    assert i18n.t("welcome.plain") == "Üdv itthon!"
    assert i18n.join(["A", "B", "C"]) == "A, B és C"
    assert i18n.clock(17, 5) == "17:05"
    assert i18n.day_and_date(datetime(2026, 10, 23, 9, 0)) == "október 23., péntek"
    assert i18n.title("Parcel left") == "Csomag érkezett"
    assert "Hungarian" in i18n.write_in()


def test_timeline_and_motion_notes_in_hungarian() -> None:
    i18n.set_language("hu")
    assert i18n.t("timeline.held", reason=i18n.reason("quiet_hours")) == "visszatartva: csendes órák"
    assert i18n.t("timeline.hushed_until", time="22:00") == "Némítva 22:00-ig"
    assert i18n.t("motion.on_late", pct=10) == "bekapcsolt 10%-on (késő éjjel)"
    assert i18n.t("night.overnight_alert", name="Kapu", time="03:12") == "Kapu 03:12-kor nyílt ki, senki sem volt fent (riasztott)"
    i18n.set_language("en")
    assert i18n.reason("no_speaker") == "no speaker"


def test_english_is_unchanged() -> None:
    assert i18n.clock(17, 30) == "5:30pm"
    assert i18n.t("welcome.plain") == "Welcome back."
    assert i18n.title("Parcel left") == "Parcel left"


def test_every_sentence_and_setting_has_hungarian() -> None:
    import json
    from pathlib import Path

    assert set(i18n.STRINGS["hu"]) == set(i18n.STRINGS["en"])
    folder = Path(__file__).parent.parent / "custom_components" / "sentient" / "translations"

    def keys(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from keys(v, f"{path}.{k}")
        else:
            yield path

    en = json.loads((folder / "en.json").read_text())
    hu = json.loads((folder / "hu.json").read_text())
    assert set(keys(hu)) == set(keys(en))


# ----- learned routines -----------------------------------------------------------------
def test_routine_and_link_described_in_hungarian() -> None:
    from custom_components.sentient.learning import Link, Routine

    i18n.set_language("hu")
    r = Routine(entity_id="light.eloszoba", name="Előszoba lámpa", state="on", minute=17 * 60 + 30,
                days="on weekdays", support=0.8, seen=8, spread=5)
    assert r.describe() == "Az előszoba lámpa hétköznap általában 17:30 körül bekapcsol."
    assert r.summary()["how_sure"] == "a napok 80%-ában (az utolsó 10 hétköznapból 8)"
    link = Link(cause="binary_sensor.hatso_ajto", cause_name="Hátsó ajtó", cause_state="on",
                entity_id="light.konyha", name="Konyha lámpa", state="on", dark=True, lag=40,
                support=0.9, seen=8, cause_class="door")
    assert link.describe() == (
        "Amikor a hátsó ajtó kinyílik sötétedés után, a konyha lámpa általában egy percen belül bekapcsol."
    )


# ----- memory -----------------------------------------------------------------------------
async def test_memory_search_copes_with_hungarian(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    memory = entry.runtime_data.memory
    hutch = memory.add("A nyúlketrec a fészer mellett van").id
    memory.add("A kút szivattyúja a garázsban van")
    memory.add("Kata szereti a kávét tejjel")

    assert memory.search("hol vannak a nyulak?")[0].id == hutch
    assert "kút" in memory.search("kut")[0].text  # without accents
    assert "kút" in memory.search("a kúthoz")[0].text  # with a case ending
    assert "kávét" in memory.search("Milyen kávét iszik Kata?")[0].text

    # a near-duplicate replaces the old one instead of piling up
    before = len(memory.items)
    memory.add("A nyúlketrec a fészer mellett van.")
    assert len(memory.items) == before


def test_memory_words_keep_accented_letters() -> None:
    from custom_components.sentient.memory import _words

    assert _words("Nyúlketrec, kút és fészer!") == {"nyulketrec", "kut", "feszer"}
    assert _words("The rabbits' hutch") == {"rabbits", "hutch"}


# ----- noticing ---------------------------------------------------------------------------
def test_hungarian_finished_remark_is_caught() -> None:
    from custom_components.sentient.notice import Change, contradicts

    start = Change(entity_id="sensor.mosogep", name="Mosógép", area="Fürdő", old="standby",
                   new="running", unit="", when=datetime.now())
    assert contradicts("A mosógép végzett.", [start])
    assert contradicts("Elkészült a kávé.", [start])
    assert not contradicts("A mosógép elindult.", [start])


# ----- security summary --------------------------------------------------------------------
def test_security_summary_in_hungarian() -> None:
    from custom_components.sentient.security import summary

    i18n.set_language("hu")
    text = summary([{"name": "Bejárati ajtó"}, {"name": "Garázskapu"}], [],
                   [{"entity_id": "alarm_control_panel.x", "name": "Riasztó", "state": "armed_night"}])
    assert text == "Bejárati ajtó és Garázskapu nyitva vannak. A riasztó élesítve (éjszakai)."
    assert summary([], [], []) == "Minden csukva van és bezárva."


# ----- spoken times --------------------------------------------------------------------------
async def test_reminder_times_in_hungarian(hass: HomeAssistant) -> None:
    from custom_components.sentient.watches import local_when

    await _setup(hass)
    now = dt_util.now()
    assert local_when(now.replace(hour=18, minute=0)) == "ma 18:00"


# ----- calendar --------------------------------------------------------------------------
async def test_calendar_event_added(hass: HomeAssistant) -> None:
    await _setup(hass)
    calls = async_mock_service(hass, "calendar", "create_event")
    hass.states.async_set("calendar.csalad", "off", {"friendly_name": "Család", "supported_features": 1})
    hass.states.async_set("calendar.unnepek", "off", {"friendly_name": "Ünnepek", "supported_features": 0})
    api = await llm.async_get_api(hass, "sentient", _ctx())

    out = await api.async_call_tool(llm.ToolInput(
        tool_name="sentient_add_calendar_event",
        tool_args={"summary": "Fogorvos", "start": "2026-10-27 15:00", "location": "Szigetvár"},
    ))
    assert out["added"] == "Fogorvos" and out["calendar"] == "Család"
    data = calls[-1].data
    assert data["entity_id"] == "calendar.csalad"
    assert data["start_date_time"] == "2026-10-27T15:00:00+01:00"  # winter time by then
    assert data["end_date_time"] == "2026-10-27T16:00:00+01:00"
    assert data["location"] == "Szigetvár"

    # all-day, over several days: the end date is exclusive
    await api.async_call_tool(llm.ToolInput(
        tool_name="sentient_add_calendar_event",
        tool_args={"summary": "Nyaralás", "start": "2026-07-01", "end": "2026-07-07"},
    ))
    assert calls[-1].data["start_date"] == "2026-07-01" and calls[-1].data["end_date"] == "2026-07-08"

    bad = await api.async_call_tool(llm.ToolInput(
        tool_name="sentient_add_calendar_event", tool_args={"summary": "x", "start": "holnap délután"},
    ))
    assert bad["error"].startswith("Nem értem")


async def test_calendar_asks_which_when_there_are_several(hass: HomeAssistant) -> None:
    await _setup(hass)
    calls = async_mock_service(hass, "calendar", "create_event")
    hass.states.async_set("calendar.csalad", "off", {"friendly_name": "Család", "supported_features": 1})
    hass.states.async_set("calendar.munka", "off", {"friendly_name": "Munka", "supported_features": 7})
    api = await llm.async_get_api(hass, "sentient", _ctx())

    out = await api.async_call_tool(llm.ToolInput(
        tool_name="sentient_add_calendar_event", tool_args={"summary": "Meeting", "start": "2026-10-27 09:00"},
    ))
    assert sorted(out["which_calendar"]) == ["Család", "Munka"] and not calls
    out = await api.async_call_tool(llm.ToolInput(
        tool_name="sentient_add_calendar_event",
        tool_args={"summary": "Meeting", "start": "2026-10-27 09:00", "calendar": "munka"},
    ))
    assert out["calendar"] == "Munka" and calls[-1].data["entity_id"] == "calendar.munka"


async def test_prompt_asks_for_hungarian(hass: HomeAssistant) -> None:
    await _setup(hass)
    api = await llm.async_get_api(hass, "sentient", _ctx())
    assert "Always speak and answer in Hungarian." in api.api_prompt
    assert any(tool.name == "sentient_add_calendar_event" for tool in api.tools)


# ----- who may do what on the sidebar page ---------------------------------------------------
async def test_non_admin_can_read_but_not_change_or_spend(
    hass: HomeAssistant, hass_ws_client, hass_read_only_access_token
) -> None:
    await _setup(hass, "en")
    ws = await hass_ws_client(hass, hass_read_only_access_token)

    await ws.send_json({"id": 1, "type": "sentient/timeline"})
    assert (await ws.receive_json())["success"]
    await ws.send_json({"id": 2, "type": "sentient/mutes"})
    assert (await ws.receive_json())["success"]

    await ws.send_json({"id": 3, "type": "sentient/mutes", "clear": True})
    assert (await ws.receive_json())["error"]["code"] == "unauthorized"
    await ws.send_json({"id": 4, "type": "sentient/look", "camera": "camera.garden", "question": "cats?"})
    assert (await ws.receive_json())["error"]["code"] == "unauthorized"
    await ws.send_json({"id": 5, "type": "sentient/digest"})
    assert (await ws.receive_json())["error"]["code"] == "unauthorized"


def test_upstream_040_sentences_in_hungarian() -> None:
    from custom_components.sentient import i18n

    i18n.set_language("hu")
    try:
        assert i18n.t("camsec.fence", place="Udvar") == "Valaki van itt: Udvar, pedig a kapu és az ajtók nem nyíltak ki"
        assert i18n.t("alert.heads_up", text="valaki a tárolónál") == "Figyelem: valaki a tárolónál."
        assert i18n.t("follow.failed", what="Spájz lámpa", verb=i18n.t("follow.on"), why="") == (
            "Sajnos a spájz lámpa nem reagált, így nem sikerült bekapcsolni."
        )
        assert i18n.t("offer.default", name="Bekapcsolni a fűtést") == "Megcsináljam: bekapcsolni a fűtést?"
    finally:
        i18n.set_language("en")
