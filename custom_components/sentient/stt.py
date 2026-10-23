"""Sentient's voice STT: speech-to-text that also works out who is speaking.

Pick it as the speech-to-text of an Assist pipeline. It passes the audio straight on to the real
speech-to-text (Home Assistant Cloud, Whisper, ... — set in Sentient's settings, or the first other one
there is) and, at the same time, sends the utterance to the Sentient Voice ID app. The words come back
exactly as the real speech-to-text heard them; who said them is known to Sentient a moment before the
conversation agent runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
import logging
from typing import Any

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
    async_get_speech_to_text_engine,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CONF_VOICE_STT_BASE, DOMAIN

_LOGGER = logging.getLogger(__name__)

MAX_SECONDS = 30  # keep at most this much audio for the voice print
ID_WAIT = 3.0  # seconds to wait for the voice print once the words are back


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    async_add_entities([SentientVoiceSTT(hass, entry)])


class SentientVoiceSTT(SpeechToTextEntity):
    _attr_has_entity_name = True
    _attr_translation_key = "voice_stt"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_voice_stt"
        self._attr_device_info = {"identifiers": {(DOMAIN, entry.entry_id)}, "name": entry.title}

    # ----- the real speech-to-text ---------------------------------------------------------------
    def _base_id(self) -> str | None:
        chosen = (self._entry.runtime_data.options or {}).get(CONF_VOICE_STT_BASE)
        if chosen and chosen != self.entity_id:
            return chosen
        others = [s.entity_id for s in self.hass.states.async_all("stt") if s.entity_id != self.entity_id]
        return "stt.home_assistant_cloud" if "stt.home_assistant_cloud" in others else (others[0] if others else None)

    def _base(self) -> Any:
        base_id = self._base_id()
        return async_get_speech_to_text_engine(self.hass, base_id) if base_id else None

    def _from_base(self, attr: str, fallback: list) -> list:
        base = self._base()
        try:
            return list(getattr(base, attr)) if base is not None else fallback
        except Exception:  # noqa: BLE001 - a base that isn't ready yet
            return fallback

    @property
    def supported_languages(self) -> list[str]:
        return self._from_base("supported_languages", [self.hass.config.language])

    @property
    def supported_formats(self) -> list[AudioFormats]:
        return self._from_base("supported_formats", [AudioFormats.WAV])

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return self._from_base("supported_codecs", [AudioCodecs.PCM])

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        return self._from_base("supported_bit_rates", [AudioBitRates.BITRATE_16])

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        return self._from_base("supported_sample_rates", [AudioSampleRates.SAMPLERATE_16000])

    @property
    def supported_channels(self) -> list[AudioChannels]:
        return self._from_base("supported_channels", [AudioChannels.CHANNEL_MONO])

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {"speech_to_text": self._base_id()}

    # ----- one utterance ---------------------------------------------------------------------------
    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        base = self._base()
        if base is None:
            _LOGGER.error("Sentient's voice STT has no speech-to-text to pass the audio to")
            return SpeechResult(None, SpeechResultState.ERROR)

        voice = self._entry.runtime_data.voice
        usable = (
            voice.enabled
            and metadata.codec == AudioCodecs.PCM
            and metadata.sample_rate == AudioSampleRates.SAMPLERATE_16000
            and metadata.bit_rate == AudioBitRates.BITRATE_16
            and metadata.channel == AudioChannels.CHANNEL_MONO
        )
        audio = bytearray()
        limit = 16000 * 2 * MAX_SECONDS
        identify: list[asyncio.Task] = []

        async def tee() -> AsyncIterable[bytes]:
            async for chunk in stream:
                if usable and len(audio) < limit:
                    audio.extend(chunk[: limit - len(audio)])
                yield chunk
            if usable:  # the speech has ended: start on the voice while the words are worked out
                identify.append(self.hass.async_create_task(voice.async_identify(bytes(audio))))

        try:
            if hasattr(base, "internal_async_process_audio_stream"):
                result = await base.internal_async_process_audio_stream(metadata, tee())
            else:  # a legacy provider
                result = await base.async_process_audio_stream(metadata, tee())
        finally:
            if identify:
                try:
                    await asyncio.wait_for(asyncio.shield(identify[0]), ID_WAIT)
                except (TimeoutError, asyncio.TimeoutError):
                    _LOGGER.debug("Voice print took too long; the conversation goes on without it")
        return result
