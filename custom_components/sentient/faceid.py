"""Who is in a camera picture — Sentient's side of the face recognition in the Sentient Voice ID app.

Cameras often recognise faces themselves (eufy, UniFi, Frigate…), but not always: the face is too far,
the integration can't name it, the camera can't do it at all. Sentient then asks the app, which runs a
small face model locally and compares the faces with the ones it has been taught. Pictures are never
stored — only the face prints.

Faces are taught by hand: from a camera right now ("Kata is at the gate now, learn her face"), from a
recent visit ("that was Sanya"), or from a photo (the sentient.learn_face action).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import CONF_VOICE_ID_URL

_LOGGER = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=15)


class FaceError(HomeAssistantError):
    """The face service said no, or couldn't be reached."""


async def async_picture(hass: HomeAssistant, entity_id: str) -> bytes:
    """The current picture of a camera, or an image entity's picture (a camera's last event snapshot)."""
    if entity_id.startswith("image."):
        from homeassistant.components.image.const import DATA_COMPONENT  # noqa: PLC0415

        component = hass.data.get(DATA_COMPONENT)
        entity = component.get_entity(entity_id) if component else None
        if entity is None:
            raise FaceError(f"no picture called {entity_id}")
        content = await entity.async_image()
        if not content:
            raise FaceError(f"{entity_id} has no picture yet")
        return content
    from homeassistant.components.camera import async_get_image  # noqa: PLC0415

    return (await async_get_image(hass, entity_id, timeout=15)).content


async def async_photo(hass: HomeAssistant, source: str) -> bytes:
    """A photo the user put in Home Assistant: a media-source id ("My media") or a path HA may read."""
    path: Path | None = None
    if source.startswith("media-source://"):
        from homeassistant.components.media_source import async_resolve_media  # noqa: PLC0415

        played = await async_resolve_media(hass, source, None)
        path = played.path
        if path is None:
            raise FaceError("that photo isn't a local file — put it in My media first")
    else:
        path = Path(source)
        if not await hass.async_add_executor_job(hass.config.is_allowed_path, str(path)):
            raise FaceError(f"Home Assistant isn't allowed to read {source}")
    try:
        return await hass.async_add_executor_job(path.read_bytes)
    except OSError as err:
        raise FaceError(f"could not read {source}: {err}") from err


class FaceID:
    def __init__(self, hass: HomeAssistant, options: dict[str, Any]) -> None:
        self.hass = hass
        self.options = options

    @property
    def url(self) -> str:
        return str(self.options.get(CONF_VOICE_ID_URL) or "").strip().rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self.enabled:
            raise FaceError("face recognition isn't set up (no Voice ID address in Sentient's settings)")
        session = async_get_clientsession(self.hass)
        try:
            async with session.request(method, f"{self.url}{path}", timeout=_TIMEOUT, **kwargs) as resp:
                text = await resp.text()
                if resp.status == 404 and "Not Found" in text:
                    raise FaceError("the Voice ID app is too old for faces — update it")
                if resp.status >= 400:
                    raise FaceError(text or f"HTTP {resp.status}")
                return json.loads(text) if text else {}
        except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as err:
            raise FaceError(f"the Voice ID app can't be reached: {err}") from err

    async def async_identify(self, picture: bytes, learn: bool = True) -> dict[str, Any] | None:
        """Who is in the picture: {"name", "maybe", "score", "id", "count"} — never raises."""
        if not self.enabled or not picture:
            return None
        try:
            res = await self._call("POST", f"/face/identify?learn={'1' if learn else '0'}", data=picture,
                                   headers={"Content-Type": "application/octet-stream"})
        except FaceError as err:
            _LOGGER.debug("Face check skipped: %s", err)
            return None
        best = res.get("best") or {}
        return {
            "name": best.get("name") if best.get("known") else None,
            "maybe": best.get("name") if best.get("unsure") else None,
            "score": best.get("score"),
            "id": best.get("id"),
            "count": int(res.get("count") or 0),
        }

    async def async_learn_picture(self, name: str, picture: bytes) -> int:
        res = await self._call("POST", "/face/enroll", params={"name": name}, data=picture,
                               headers={"Content-Type": "application/octet-stream"})
        return int(res.get("samples") or 0)

    async def async_learn_ids(self, name: str, ids: list[str]) -> int:
        res = await self._call("POST", "/face/enroll", json={"name": name, "ids": ids})
        return int(res.get("samples") or 0)

    async def async_people(self) -> list[dict[str, Any]]:
        return await self._call("GET", "/face/people")

    async def async_forget(self, name: str) -> bool:
        try:
            await self._call("DELETE", f"/face/people/{name}")
        except FaceError as err:
            if "no face called" in str(err):
                return False
            raise
        return True
