"""Semantic memory: finding memories by meaning, with a stand-in for the app's text model."""

import math
from unittest.mock import patch

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sentient import i18n
from custom_components.sentient.const import DOMAIN
from custom_components.sentient.semantic import EmbedError

# A tiny "model": each word belongs to a concept; a sentence is the sum of its concepts plus a shared part,
# so unrelated sentences score about 0.72 and related ones 0.85+ (as multilingual-e5 does).
CONCEPTS = {
    "dog": ["kutya", "kutyus", "bodri"],
    "food": ["eszik", "enni", "táp", "tápot", "kaja", "etetni"],
    "coffee": ["kávé", "kávét", "kávéját", "tejjel", "iszik", "issza"],
    "story": ["mese", "mesét", "meséje", "sárkány", "sárkányos", "tovább", "meséld"],
    "bins": ["kuka", "kukát", "szemét", "szemetet"],
    "kata": ["kata"],
    "szofi": ["szofi", "szofinak"],
}
DIMS = ["_"] + list(CONCEPTS)


def _vector(text: str) -> list[float]:
    v = [0.0] * len(DIMS)
    v[0] = 1.6
    for word in text.lower().replace("?", " ").replace(".", " ").replace(":", " ").replace(",", " ").split():
        for i, (_c, words) in enumerate(CONCEPTS.items(), start=1):
            if word in words:
                v[i] = 1.0
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class FakeModel:
    def __init__(self) -> None:
        self.model = "multilingual-e5-small"
        self.down = False
        self.calls: list[tuple[str, int]] = []

    async def embed(self, texts, kind):
        if self.down:
            raise EmbedError("the text model can't be reached")
        self.calls.append((kind, len(texts)))
        return self.model, [_vector(t) for t in texts]


MEMORIES = [
    "Bodri csak a száraz tápot szereti",
    "Kata kávét iszik tejjel",
    "Szofi sárkányos meséje: a sárkány megtalálta a kulcsot, holnap hazarepül",
    "A kukát csütörtök este visszük ki",
]


@pytest.fixture(autouse=True)
def _english_afterwards():
    yield
    i18n.set_language("en")


async def _setup(hass: HomeAssistant, model: FakeModel, url: str = "http://app:8765"):
    hass.config.language = "hu"
    entry = MockConfigEntry(domain=DOMAIN, title="Nova", unique_id=DOMAIN, data={"persona": "Nova", "speakers": []},
                            options={"embed_url": url})
    entry.add_to_hass(hass)
    with patch("custom_components.sentient.semantic.Embedder.async_embed", model.embed):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    memory = entry.runtime_data.memory
    for text in MEMORIES:
        memory.add(text)
    return entry, memory


async def test_finds_by_meaning_where_words_dont_match(hass: HomeAssistant) -> None:
    model = FakeModel()
    entry, memory = await _setup(hass, model)
    with patch("custom_components.sentient.semantic.Embedder.async_embed", model.embed):
        assert await memory.semantic.async_sync() == 4
        assert memory.semantic.coverage() == (4, 4)

        assert memory.search("Mit eszik a kutya?", count_hits=False) == []  # no word in common
        found = await memory.async_search("Mit eszik a kutya?", count_hits=False)
        assert found[0].text.startswith("Bodri") and len(found) == 1

        story = await memory.async_search("Meséld tovább Szofinak a mesét", count_hits=False)
        assert story[0].text.startswith("Szofi sárkányos")

        # through the voice assistant's tool
        ctx = llm.LLMContext(platform="t", context=None, language="hu", assistant="conversation", device_id=None)
        api = await llm.async_get_api(hass, "sentient", ctx)
        out = await api.async_call_tool(llm.ToolInput(tool_name="sentient_recall", tool_args={"query": "mit iszik Kata reggel"}))
        assert out["memories"][0]["text"] == "Kata kávét iszik tejjel"


async def test_vectors_follow_edits_and_forgetting(hass: HomeAssistant) -> None:
    model = FakeModel()
    entry, memory = await _setup(hass, model)
    with patch("custom_components.sentient.semantic.Embedder.async_embed", model.embed):
        await memory.semantic.async_sync()
        model.calls.clear()
        assert await memory.semantic.async_sync() == 0 and not model.calls  # nothing new: no model call

        bins = next(m for m in memory.items if "kukát" in m.text)
        memory.update(bins.id, "A szemetet kedden visszük ki")
        assert await memory.semantic.async_sync() == 1
        memory.forget(bins.id)
        await memory.semantic.async_sync()
        assert bins.id not in memory.semantic.vectors

        # a new model in the app: the next search notices, and everything is worked out again,
        # so old and new vectors are never compared (that search falls back to keywords)
        model.model = "another-model"
        assert [m.text for m in await memory.async_search("kávét", count_hits=False)] == ["Kata kávét iszik tejjel"]
        assert await memory.semantic.async_sync() == 3
        assert memory.semantic.model == "another-model"


async def test_keywords_when_the_app_is_away(hass: HomeAssistant) -> None:
    model = FakeModel()
    entry, memory = await _setup(hass, model)
    with patch("custom_components.sentient.semantic.Embedder.async_embed", model.embed):
        await memory.semantic.async_sync()
        model.down = True
        found = await memory.async_search("kávét", count_hits=False)
        assert [m.text for m in found] == ["Kata kávét iszik tejjel"]
        assert memory.semantic.vectors  # the vectors it had are kept for when it's back


async def test_forgetting_asks_when_unsure(hass: HomeAssistant) -> None:
    model = FakeModel()
    entry, memory = await _setup(hass, model)
    memory.add("Kata kávéját cukor nélkül issza")
    with patch("custom_components.sentient.semantic.Embedder.async_embed", model.embed):
        await memory.semantic.async_sync()
        removed, closest = await memory.async_forget("amit reggelente iszik")
        assert removed is None
        assert {m.text for m in closest} >= {"Kata kávét iszik tejjel", "Kata kávéját cukor nélkül issza"}

        removed, _ = await memory.async_forget(closest[0].id)
        assert removed is not None and removed.id not in {m.id for m in memory.items}


async def test_off_without_an_address(hass: HomeAssistant) -> None:
    model = FakeModel()
    entry, memory = await _setup(hass, model, url="")
    with patch("custom_components.sentient.semantic.Embedder.async_embed", model.embed):
        assert await memory.semantic.async_sync() == 0
        assert not model.calls
        assert [m.text for m in await memory.async_search("kávét", count_hits=False)] == ["Kata kávét iszik tejjel"]
