"""The dashboard cards register themselves as Lovelace resources and follow the version."""

from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

from custom_components.sentient import _async_card_resources


async def test_cards_become_resources_and_move_with_the_version(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "lovelace", {})
    from homeassistant.components.lovelace.const import LOVELACE_DATA

    res = hass.data[LOVELACE_DATA].resources
    await _async_card_resources(hass, "1.0.0")
    urls = sorted(i["url"] for i in res.async_items())
    assert urls == ["/sentient_frontend/sentient-hub-card.js?v=1.0.0", "/sentient_frontend/sentient-shows-card.js?v=1.0.0"]
    await _async_card_resources(hass, "1.0.1")
    urls = sorted(i["url"] for i in res.async_items())
    assert urls == ["/sentient_frontend/sentient-hub-card.js?v=1.0.1", "/sentient_frontend/sentient-shows-card.js?v=1.0.1"]
