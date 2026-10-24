"""The voice ID app's logic, with a stand-in for the speech model (the real one runs in the app)."""

import importlib.util
import io
from pathlib import Path
import wave

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

SERVER = Path(__file__).parent.parent / "voice_id" / "server.py"
_spec = importlib.util.spec_from_file_location("voice_id_server", SERVER)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)

# Each "person" has a voice direction; the fake model returns it plus a little noise.
_VOICES = {k: server.unit(np.random.default_rng(k).normal(size=64)) for k in range(1, 6)}


def _embed(x: np.ndarray) -> np.ndarray:
    who = int(round(float(np.max(np.abs(x))) * 10))  # the test audio's loudness says who is talking
    noise = np.random.default_rng(len(x)).normal(scale=0.03, size=64)
    return _VOICES[who] + noise


def _speech(who: int, seconds: float = 1.5, wav: bool = False) -> bytes:
    n = int(seconds * 16000)
    x = np.concatenate([np.zeros(4000), np.sin(np.arange(n) / 7.0) * who / 10, np.zeros(4000)])
    pcm = (x * 32767).astype("<i2").tobytes()
    if not wav:
        return pcm
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    return out.getvalue()


async def test_enrol_identify_and_forget(tmp_path, socket_enabled) -> None:
    svc = server.Service(server.Voices(tmp_path), _embed)
    async with TestClient(TestServer(svc.app())) as client:
        for name, who in (("Sanya", 1), ("Kata", 2), ("Lili", 3)):
            for seconds in (1.2, 1.6, 2.1):
                res = await client.post(f"/enroll?name={name}", data=_speech(who, seconds))
                assert res.status == 200

        res = await (await client.post("/identify", data=_speech(3, 1.8, wav=True))).json()
        assert res["name"] == "Lili" and res["known"] and not res["unsure"]

        # someone it was never taught is not given a family member's name
        res = await (await client.post("/identify", data=_speech(5, 1.4))).json()
        assert not res["known"]

        # teach a voice it has just heard, by the print's id
        heard = await (await client.post("/identify", data=_speech(4, 1.7))).json()
        res = await (await client.post("/enroll", json={"name": "Nagyi", "ids": [heard["id"]]})).json()
        assert res == {"name": "Nagyi", "samples": 1}

        people = {p["name"]: p for p in await (await client.get("/people")).json()}
        assert set(people) == {"Sanya", "Kata", "Lili", "Nagyi"} and people["Sanya"]["samples"] == 3

        assert (await client.post("/identify", data=b"\0\0" * 3000)).status == 422  # too little speech
        assert (await client.delete("/people/kata")).status == 200
        assert (await client.delete("/people/kata")).status == 404

    # voice prints survive a restart; audio is never written
    again = server.Voices(tmp_path)
    assert set(again.people) == {"Sanya", "Lili", "Nagyi"}
    assert [p.name for p in tmp_path.iterdir()] == ["voices.json"]


async def test_embed_endpoint(tmp_path, socket_enabled) -> None:
    seen = []

    def embed_texts(texts, kind):
        seen.append(kind)
        return np.array([[1.0, 0.0] for _ in texts])

    svc = server.Service(server.Voices(tmp_path), _embed, server.Meaning(embed_texts))
    async with TestClient(TestServer(svc.app())) as client:
        res = await (await client.post("/embed", json={"texts": ["a kút", "kávé"], "kind": "query"})).json()
        assert res["model"] == "multilingual-e5-small" and res["vectors"] == [[1.0, 0.0], [1.0, 0.0]]
        assert (await (await client.post("/embed", json={"texts": []})).json())["vectors"] == []
        assert (await client.post("/embed", data=b"nope")).status == 400
    assert seen == ["query"]

    # without a text model the endpoint isn't there, and voices still work
    async with TestClient(TestServer(server.Service(server.Voices(tmp_path), _embed).app())) as client:
        assert (await client.post("/embed", json={"texts": ["x"]})).status == 404


def test_silence_is_trimmed() -> None:
    x = np.concatenate([np.zeros(16000), np.ones(8000) * 0.3, np.zeros(16000)]).astype(np.float32)
    assert 8000 <= len(server.trim(x)) <= 8000 + 2 * 1600
