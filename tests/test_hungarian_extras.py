"""Hungarian additions on top of the 0.53 features: the voice guard, appliances, bedtime and the AI prompt language."""

from custom_components.sentient import i18n
from custom_components.sentient.guard import BROAD, _named, _words


def test_guard_understands_hungarian_endings_and_all_words() -> None:
    said = _words("Kapcsold le a konyhai lámpát")
    assert _named("Konyhai lámpa", said)
    assert not _named("Nappali lámpa", said)
    assert _named("Shed light", _words("turn off the shed lights"))
    assert BROAD.search("kapcsold le az összes lámpát")
    assert not BROAD.search("kapcsold le a lámpát")


def test_new_sentences_in_hungarian() -> None:
    i18n.set_language("hu")
    try:
        assert i18n.t("appliance.finished", name="Mosógép") == "Mosógép végzett."
        assert i18n.t("visitors.just_someone_at_place", place="Kapu") == "Kapu: valaki jött."
        assert i18n.t("follow.while_away") == "Amíg nem voltál itt: "
        assert "fagyot" in i18n.t("foresight.frost")
    finally:
        i18n.set_language("en")


def test_every_ai_prompt_asks_for_the_house_language() -> None:
    from custom_components.sentient.ai import _in_house_language

    once = _in_house_language({"instructions": "Describe the drive."})
    assert once["instructions"].endswith(i18n.write_in())
    assert _in_house_language(once)["instructions"] == once["instructions"]  # not twice


def test_bedtime_routine_names_in_hungarian() -> None:
    from custom_components.sentient.sleep import Sleep

    sleep = Sleep.__new__(Sleep)
    sleep.options = {}
    assert sleep.is_bedtime_routine("Jó éjt")
    assert sleep.is_bedtime_routine("Éjszakai mód")
    assert not sleep.is_bedtime_routine("Reggeli kávé")
