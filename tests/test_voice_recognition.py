"""Sentient's side of voice recognition: the voice STT, who is talking, and learning voices."""

from unittest.mock import patch

import pytest

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    async_get_speech_to_text_entity,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sentient import i18n
from custom_components.sentient.const import DOMAIN
from custom_components.sentient.voice import VoiceIDError

META = SpeechMetadata(language="hu-HU", format=AudioFormats.WAV, codec=AudioCodecs.PCM,
                      bit_rate=AudioBitRates.BITRATE_16, sample_rate=AudioSampleRates.SAMPLERATE_16000,
                      channel=AudioChannels.CHANNEL_MONO)
SECOND = b"\x10\x00" * 16000


class FakeBase:
    """The real speech-to-text the voice STT passes the audio to."""

    supported_languages = ["hu-HU", "en-US"]
    supported_formats = [AudioFormats.WAV]
    supported_codecs = [AudioCodecs.PCM]
    supported_bit_rates = [AudioBitRates.BITRATE_16]
    supported_sample_rates = [AudioSampleRates.SAMPLERATE_16000]
    supported_channels = [AudioChannels.CHANNEL_MONO]

    def __init__(self, text: str) -> None:
        self.text, self.heard = text, 0

    async def internal_async_process_audio_stream(self, metadata, stream):
        self.language = metadata.language
        async for chunk in stream:
            self.heard += len(chunk)
        return SpeechResult(self.text, SpeechResultState.SUCCESS)


class FakeApp:
    """The Sentient Voice ID app."""

    def __init__(self) -> None:
        self.next = {"name": "Lili", "known": True, "unsure": False, "score": 0.82, "id": "p1"}
        self.calls: list[tuple] = []
        self.down = False

    async def call(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs.get("json")))
        if self.down:
            raise VoiceIDError("the Voice ID app can't be reached")
        if path.startswith("/identify"):
            return dict(self.next)
        if path == "/enroll":
            return {"name": kwargs["json"]["name"], "samples": 1}
        if path == "/people":
            return [{"name": "Lili", "samples": 5, "learned": 2}]
        return {"forgotten": path.rsplit("/", 1)[-1]}


@pytest.fixture(autouse=True)
def _english_afterwards():
    yield
    i18n.set_language("en")


async def _setup(hass: HomeAssistant, url: str = "http://voice-id:8765") -> MockConfigEntry:
    hass.config.language = "hu"
    entry = MockConfigEntry(domain=DOMAIN, title="Nova", unique_id=DOMAIN, data={"persona": "Nova", "speakers": []},
                            options={"voice_id_url": url, "voice_stt_base": "stt.real_speech_to_text"})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def _stt(hass: HomeAssistant, entry):
    entity_id = er.async_get(hass).async_get_entity_id("stt", DOMAIN, f"{entry.entry_id}_voice_stt")
    return entity_id, async_get_speech_to_text_entity(hass, entity_id)


async def _say(entity, seconds: int = 2):
    async def stream():
        for _ in range(seconds):
            yield SECOND
    return await entity.async_process_audio_stream(META, stream())


async def _prompt(hass: HomeAssistant) -> llm.APIInstance:
    ctx = llm.LLMContext(platform="test", context=None, language="hu", assistant="conversation", device_id=None)
    return await llm.async_get_api(hass, "sentient", ctx)


async def test_words_pass_through_and_the_speaker_is_known(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    app, base = FakeApp(), FakeBase("mesélj tovább")
    with patch("custom_components.sentient.voice.VoiceID._call", app.call), \
         patch("custom_components.sentient.stt.async_get_speech_to_text_engine", return_value=base):
        entity_id, stt = _stt(hass, entry)
        assert stt.supported_languages == ["hu-HU", "en-US", "hu", "en"]
        result = await _say(stt)
        await hass.async_block_till_done()

        assert result.text == "mesélj tovább" and base.heard == 2 * len(SECOND)  # every byte went on
        assert app.calls[0][1] == "/identify"
        assert hass.states.get("sensor.nova_beszelo").state == "Lili"
        api = await _prompt(hass)
        assert "The person talking to you is Lili (recognised by their voice)" in api.api_prompt
        assert {"sentient_learn_voice", "sentient_voices", "sentient_forget_voice"} <= {t.name for t in api.tools}

        # a voice it isn't sure about is only a guess
        app.next = {"name": "Kata", "known": False, "unsure": True, "score": 0.65, "id": "p2"}
        await _say(stt)
        assert "might be Kata" in (await _prompt(hass)).api_prompt
        assert hass.states.get("sensor.nova_beszelo").state == "unknown"


async def test_learning_a_voice(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    app, base = FakeApp(), FakeBase("tanuld meg a hangomat, Sanya vagyok")
    app.next = {"name": None, "known": False, "unsure": False, "score": 0.0, "id": "s1"}
    with patch("custom_components.sentient.voice.VoiceID._call", app.call), \
         patch("custom_components.sentient.stt.async_get_speech_to_text_engine", return_value=base):
        _id, stt = _stt(hass, entry)
        await _say(stt)
        api = await _prompt(hass)
        out = await api.async_call_tool(llm.ToolInput(tool_name="sentient_learn_voice",
                                                      tool_args={"name": "Sanya", "this_is_them": True}))
        assert out["already_heard"] == 1 and out["sentences_still_needed"] == 4
        assert ("POST", "/enroll", {"name": "Sanya", "ids": ["s1"]}) in app.calls

        # the next sentences are taken as Sanya's, without the app guessing or keeping them on its own
        app.next = {"name": "Kata", "known": False, "unsure": False, "score": 0.3, "id": "s2"}
        await _say(stt)
        assert ("POST", "/identify?learn=0", None) in app.calls
        assert ("POST", "/enroll", {"name": "Sanya", "ids": ["s2"]}) in app.calls
        assert "You are learning Sanya's voice: 3 more" in (await _prompt(hass)).api_prompt

        # learning someone else's voice doesn't take the parent's sentence
        app.calls.clear()
        out = await api.async_call_tool(llm.ToolInput(tool_name="sentient_learn_voice",
                                                      tool_args={"name": "Lili", "this_is_them": False}))
        assert out["already_heard"] == 0 and not [c for c in app.calls if c[1] == "/enroll"]

        voices = await api.async_call_tool(llm.ToolInput(tool_name="sentient_voices", tool_args={}))
        assert voices["voices"][0]["name"] == "Lili"


async def test_speech_goes_on_when_the_app_is_down(hass: HomeAssistant) -> None:
    entry = await _setup(hass)
    app, base = FakeApp(), FakeBase("kapcsold fel a lámpát")
    app.down = True
    with patch("custom_components.sentient.voice.VoiceID._call", app.call), \
         patch("custom_components.sentient.stt.async_get_speech_to_text_engine", return_value=base):
        _id, stt = _stt(hass, entry)
        result = await _say(stt)
        assert result.text == "kapcsold fel a lámpát"
        assert "couldn't tell from the voice" in (await _prompt(hass)).api_prompt


async def test_off_without_an_address(hass: HomeAssistant) -> None:
    entry = await _setup(hass, url="")
    app, base = FakeApp(), FakeBase("szia")
    with patch("custom_components.sentient.voice.VoiceID._call", app.call), \
         patch("custom_components.sentient.stt.async_get_speech_to_text_engine", return_value=base):
        _id, stt = _stt(hass, entry)
        assert (await _say(stt)).text == "szia"
        assert not app.calls  # nothing sent anywhere
        api = await _prompt(hass)
        assert "voice" not in {t.name.split("_")[-1] for t in api.tools}
        assert "from the voice" not in api.api_prompt


async def test_a_short_language_code_reaches_the_base_in_its_own_spelling(hass: HomeAssistant) -> None:
    """A pipeline set to "hu" (as a Whisper wants) still works when the base is a cloud that knows "hu-HU"."""
    import dataclasses

    entry = await _setup(hass)
    app, base = FakeApp(), FakeBase("jó reggelt")
    with patch("custom_components.sentient.voice.VoiceID._call", app.call), \
         patch("custom_components.sentient.stt.async_get_speech_to_text_engine", return_value=base):
        _entity_id, stt = _stt(hass, entry)
        short = dataclasses.replace(META, language="hu")
        assert stt.check_metadata(short)

        async def audio():
            yield SECOND

        result = await stt.async_process_audio_stream(short, audio())
    assert result.text == "jó reggelt" and base.language == "hu-HU"
