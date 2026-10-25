"""Sentient Voice ID — who in the house is speaking, worked out locally from a few seconds of speech.

A small HTTP service for the Sentient integration. Every request carries one utterance (16 kHz mono
16-bit PCM, or a WAV file). It turns the speech into a voice print (a WeSpeaker embedding) and compares
it with the voices it has been taught. Audio is never stored — only the voice prints.

    GET    /health                      -> {"ok": true, "people": 3}
    POST   /identify[?learn=0]          -> {"name": "Lili", "score": 0.81, "second": 0.52, "known": true,
                                            "unsure": false, "id": "<voice print id>"}
    POST   /enroll?name=Lili            -> {"name": "Lili", "samples": 4}   (audio in the body)
    POST   /enroll  {"name": "Lili", "ids": ["<voice print id>", ...]}      (prints it has just heard)
    GET    /people                      -> [{"name": "Lili", "samples": 4, "learned": 12}]
    DELETE /people/<name>               -> {"forgotten": "Lili"}

"known" means it is sure; "unsure" means it is probably that person but not sure enough to say.
When it is very sure, the print is kept (up to a limit) so it keeps getting better at that voice.

Faces work the same way, from a picture (JPEG/PNG in the body); pictures are never stored either:

    POST   /face/identify[?learn=0]     -> {"faces": [{"name", "score", "known", "unsure", "id", "box",
                                            "quality"}, ...], "best": <the known face, or the biggest>}
    POST   /face/enroll?name=Lili       -> {"name": "Lili", "samples": 3}   (picture in the body)
    POST   /face/enroll {"name": "Lili", "ids": ["<face id>"]}             (a face it has just seen)
    GET    /face/people  ·  DELETE /face/people/<name>
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import io
import json
import logging
import os
from pathlib import Path
import uuid
import wave

from aiohttp import web
import numpy as np

_LOGGER = logging.getLogger("voice_id")

DATA = Path(os.environ.get("VOICE_ID_DATA", "/data"))
MODEL = os.environ.get("VOICE_ID_MODEL", "/models/wespeaker.onnx")
TEXT_MODEL = os.environ.get("VOICE_ID_TEXT_MODEL", "/models/e5")
FACE_MODELS = os.environ.get("VOICE_ID_FACE_MODELS", "/models/faces")
RATE = 16000
MIN_SECONDS = 0.6
MAX_ENROLLED = 40
MAX_LEARNED = 30
RECENT = 100  # voice prints kept in memory, so "learn the voice you just heard" works
TOP = 3  # a person's score is the mean of their TOP closest prints


# ----- audio --------------------------------------------------------------------------------
def to_samples(body: bytes) -> np.ndarray:
    """16 kHz mono float samples from raw 16-bit PCM or a WAV file."""
    if body[:4] == b"RIFF":
        with wave.open(io.BytesIO(body)) as w:
            if w.getsampwidth() != 2:
                raise ValueError("only 16-bit audio")
            rate, channels = w.getframerate(), w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        x = pcm.astype(np.float32) / 32768.0
        if rate != RATE:
            n = int(len(x) * RATE / rate)
            x = np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x).astype(np.float32)
        return x
    if len(body) % 2:
        body = body[:-1]
    return np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0


def trim(x: np.ndarray) -> np.ndarray:
    """Cut the silence before and after the speech (satellites send some of both)."""
    frame = RATE // 50  # 20 ms
    n = len(x) // frame
    if n < 3:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    loud = np.where(rms > max(rms.max() * 0.08, 1e-4))[0]
    if not len(loud):
        return x[:0]
    start, end = max(0, loud[0] - 5), min(n, loud[-1] + 6)  # keep 100 ms either side
    return x[start * frame:end * frame]


# ----- voices --------------------------------------------------------------------------------
def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


class Voices:
    """A gallery of named prints — voice prints, and (as faces.json) face prints."""

    def __init__(self, folder: Path, threshold: float = 0.7, margin: float = 0.06, learn: bool = True,
                 filename: str = "voices.json") -> None:
        self.path = folder / filename
        self.threshold, self.margin, self.learn = threshold, margin, learn
        self.people: dict[str, dict[str, list[np.ndarray]]] = {}
        self.recent: OrderedDict[str, np.ndarray] = OrderedDict()
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as err:
            _LOGGER.error("Could not read %s: %s", self.path, err)
            return
        for name, p in raw.items():
            self.people[name] = {k: [unit(v) for v in p.get(k, [])] for k in ("enrolled", "learned")}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {n: {k: [[round(float(f), 6) for f in v] for v in p[k]] for k in ("enrolled", "learned")}
             for n, p in self.people.items()}
        ))
        tmp.replace(self.path)

    def remember(self, emb: np.ndarray) -> str:
        key = uuid.uuid4().hex[:12]
        self.recent[key] = emb
        while len(self.recent) > RECENT:
            self.recent.popitem(last=False)
        return key

    def scores(self, emb: np.ndarray) -> dict[str, float]:
        out = {}
        for name, p in self.people.items():
            prints = p["enrolled"] + p["learned"]
            if prints:
                sims = sorted((float(emb @ v) for v in prints), reverse=True)
                out[name] = float(np.mean(sims[:TOP]))
        return out

    def identify(self, emb: np.ndarray, learn: bool = True) -> dict:
        key = self.remember(emb)
        ranked = sorted(self.scores(emb).items(), key=lambda kv: kv[1], reverse=True)
        if not ranked:
            return {"name": None, "score": 0.0, "second": 0.0, "known": False, "unsure": False, "id": key}
        (name, best), second = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
        clear = best - second >= self.margin
        known = best >= self.threshold and clear
        unsure = not known and best >= self.threshold - 0.08 and clear
        if known and learn and self.learn and best >= self.threshold + 0.08:
            learned = self.people[name]["learned"]
            learned.append(emb)
            del learned[:-MAX_LEARNED]
            self.save()
        return {"name": name, "score": round(best, 3), "second": round(second, 3), "known": known,
                "unsure": unsure, "id": key}

    def enroll(self, name: str, embs: list[np.ndarray]) -> int:
        p = self.people.setdefault(name, {"enrolled": [], "learned": []})
        p["enrolled"].extend(embs)
        del p["enrolled"][:-MAX_ENROLLED]
        self.save()
        return len(p["enrolled"])

    def forget(self, name: str) -> bool:
        key = next((n for n in self.people if n.lower() == name.lower()), None)
        if key is None:
            return False
        del self.people[key]
        self.save()
        return True

    def listing(self) -> list[dict]:
        return [{"name": n, "samples": len(p["enrolled"]), "learned": len(p["learned"])} for n, p in self.people.items()]


# ----- faces --------------------------------------------------------------------------------
MIN_FACE = 32  # pixels; smaller faces are too blurry to tell apart
MIN_FACE_SCORE = 0.75
MAX_SIDE = 1600


class FaceModel:
    """YuNet finds the faces in a picture; SFace turns each one into 128 numbers.

    Both are small OpenCV models that run on the CPU. Loaded on first use.
    """

    def __init__(self, detector: str, recognizer: str) -> None:
        self.detector_path, self.recognizer_path = detector, recognizer
        self._detector = None
        self._recognizer = None

    def _load(self) -> None:
        import cv2  # noqa: PLC0415

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(self.detector_path, "", (320, 320), MIN_FACE_SCORE, 0.3, 50)
        self._recognizer = cv2.FaceRecognizerSF.create(self.recognizer_path, "")
        _LOGGER.info("Face models loaded")

    def __call__(self, picture: bytes) -> list[dict]:
        """Every usable face in a JPEG/PNG, biggest first: {"box": [x, y, w, h], "score", "emb"}."""
        if self._detector is None:
            self._load()
        cv2 = self._cv2
        img = cv2.imdecode(np.frombuffer(picture, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("not a picture")
        h, w = img.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]
        self._detector.setInputSize((w, h))
        _, found = self._detector.detect(img)
        faces = []
        for row in found if found is not None else []:
            x, y, fw, fh, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[-1])
            if min(fw, fh) < MIN_FACE or score < MIN_FACE_SCORE:
                continue
            crop = self._recognizer.alignCrop(img, row)
            emb = unit(self._recognizer.feature(crop).flatten())
            faces.append({"box": [round(x / scale), round(y / scale), round(fw / scale), round(fh / scale)],
                          "score": round(score, 3), "emb": emb})
        faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
        return faces


class Faces:
    """POST /face/identify (a picture) -> who is in it; POST /face/enroll; GET /face/people; DELETE."""

    def __init__(self, gallery: Voices, model) -> None:
        self.gallery, self.model = gallery, model
        self._lock = asyncio.Lock()

    async def _faces(self, request: web.Request) -> list[dict]:
        body = await request.read()
        if not body:
            raise web.HTTPBadRequest(text="a picture is needed")
        async with self._lock:
            try:
                return await asyncio.get_running_loop().run_in_executor(None, self.model, body)
            except ValueError as err:
                raise web.HTTPBadRequest(text=str(err)) from err

    async def identify(self, request: web.Request) -> web.Response:
        learn = request.query.get("learn", "1") != "0"
        out = []
        for face in await self._faces(request):
            result = self.gallery.identify(face["emb"], learn=learn)
            out.append({**result, "box": face["box"], "quality": face["score"]})
        known = [f for f in out if f["known"]]
        best = known[0] if known else (out[0] if out else None)
        return web.json_response({"faces": out, "best": best, "count": len(out)})

    async def enroll(self, request: web.Request) -> web.Response:
        if request.content_type == "application/json":
            data = await request.json()
            name = str(data.get("name") or "").strip()
            embs = [self.gallery.recent[i] for i in data.get("ids") or [] if i in self.gallery.recent]
            if not embs:
                raise web.HTTPNotFound(text="none of those faces are still remembered")
        else:
            name = request.query.get("name", "").strip()
            faces = await self._faces(request)
            if not faces:
                raise web.HTTPUnprocessableEntity(text="no clear face in the picture")
            embs = [faces[0]["emb"]]  # the biggest face is the one meant
        if not name:
            raise web.HTTPBadRequest(text="a name is needed")
        return web.json_response({"name": name, "samples": self.gallery.enroll(name, embs)})

    async def people(self, _request: web.Request) -> web.Response:
        return web.json_response(self.gallery.listing())

    async def forget(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if not self.gallery.forget(name):
            raise web.HTTPNotFound(text=f"no face called {name}")
        return web.json_response({"forgotten": name})

    def routes(self) -> list:
        return [
            web.post("/face/identify", self.identify),
            web.post("/face/enroll", self.enroll),
            web.get("/face/people", self.people),
            web.delete("/face/people/{name}", self.forget),
        ]


# ----- web ----------------------------------------------------------------------------------
class Service:
    def __init__(self, voices: Voices, embed, meaning: "Meaning | None" = None, faces: "Faces | None" = None) -> None:
        self.voices, self.embed, self.meaning, self.faces = voices, embed, meaning, faces
        self._lock = asyncio.Lock()  # one embedding at a time keeps a small box responsive

    async def _print(self, request: web.Request) -> np.ndarray:
        body = await request.read()
        try:
            x = trim(to_samples(body))
        except (ValueError, wave.Error, EOFError) as err:
            raise web.HTTPBadRequest(text=f"unreadable audio: {err}") from err
        if len(x) < MIN_SECONDS * RATE:
            raise web.HTTPUnprocessableEntity(text="too little speech")
        async with self._lock:
            return unit(await asyncio.get_running_loop().run_in_executor(None, self.embed, x))

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "people": len(self.voices.people)})

    async def identify(self, request: web.Request) -> web.Response:
        emb = await self._print(request)
        return web.json_response(self.voices.identify(emb, learn=request.query.get("learn", "1") != "0"))

    async def enroll(self, request: web.Request) -> web.Response:
        if request.content_type == "application/json":
            data = await request.json()
            name = str(data.get("name") or "").strip()
            embs = [self.voices.recent[i] for i in data.get("ids") or [] if i in self.voices.recent]
            if not embs:
                raise web.HTTPNotFound(text="none of those voice prints are still remembered")
        else:
            name = request.query.get("name", "").strip()
            embs = [await self._print(request)]
        if not name:
            raise web.HTTPBadRequest(text="a name is needed")
        return web.json_response({"name": name, "samples": self.voices.enroll(name, embs)})

    async def people(self, _request: web.Request) -> web.Response:
        return web.json_response(self.voices.listing())

    async def forget(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if not self.voices.forget(name):
            raise web.HTTPNotFound(text=f"no voice called {name}")
        return web.json_response({"forgotten": name})

    def app(self) -> web.Application:
        app = web.Application(client_max_size=8 * 1024 * 1024)
        app.add_routes([
            web.get("/health", self.health),
            web.post("/identify", self.identify),
            web.post("/enroll", self.enroll),
            web.get("/people", self.people),
            web.delete("/people/{name}", self.forget),
        ])
        if self.meaning is not None:
            app.add_routes([web.post("/embed", self.meaning.embed)])
        if self.faces is not None:
            app.add_routes(self.faces.routes())
        return app


class TextModel:
    """multilingual-e5 (base): sentences in any language (Hungarian included) to numbers for meaning.

    Loaded on first use. e5 wants "query: " in front of what is searched for and "passage: " in front of
    what is searched in; vectors are the mean over tokens, normalised.
    """

    def __init__(self, folder: str) -> None:
        self.folder = Path(folder)
        self._session = None
        self._tokenizer = None
        self._inputs: list[str] = []

    def _load(self) -> None:
        import onnxruntime as ort  # noqa: PLC0415
        from tokenizers import Tokenizer  # noqa: PLC0415

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(str(self.folder / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self._inputs = [i.name for i in self._session.get_inputs()]
        tok = Tokenizer.from_file(str(self.folder / "tokenizer.json"))
        tok.enable_truncation(max_length=256)
        tok.enable_padding(pad_id=tok.token_to_id("<pad>") or 1, pad_token="<pad>")
        self._tokenizer = tok
        _LOGGER.info("Text model loaded (%s)", ", ".join(self._inputs))

    def __call__(self, texts: list[str], kind: "str | list[str]") -> np.ndarray:
        if self._session is None:
            self._load()
        kinds = kind if isinstance(kind, list) else [kind] * len(texts)
        enc = self._tokenizer.encode_batch(
            [("query: " if k == "query" else "passage: ") + t for t, k in zip(texts, kinds)]
        )
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, {k: v for k, v in feeds.items() if k in self._inputs})[0]
        m = mask[..., None].astype(np.float32)
        vecs = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


class Meaning:
    """POST /embed {"texts": [...], "kind": "query"|"passage"} -> {"model": ..., "vectors": [[...], ...]}"""

    MAX_TEXTS = 64

    def __init__(self, embed_texts, model_id: str = "multilingual-e5-base") -> None:
        self.embed_texts = embed_texts
        self.MODEL_ID = model_id  # Sentient works every memory out again when this changes
        self._lock = asyncio.Lock()

    async def embed(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except ValueError as err:
            raise web.HTTPBadRequest(text="JSON needed") from err
        texts = [str(t) for t in data.get("texts") or []][: self.MAX_TEXTS]
        kind = "query" if data.get("kind") == "query" else "passage"
        if isinstance(data.get("kinds"), list) and len(data["kinds"]) >= len(texts):  # one per text
            kind = ["query" if k == "query" else "passage" for k in data["kinds"][: len(texts)]]
        if not texts:
            return web.json_response({"model": self.MODEL_ID, "vectors": []})
        async with self._lock:
            vecs = await asyncio.get_running_loop().run_in_executor(None, self.embed_texts, texts, kind)
        return web.json_response({"model": self.MODEL_ID, "vectors": [[round(float(f), 5) for f in v] for v in vecs]})


def make_embedder(model: str):
    import sherpa_onnx  # noqa: PLC0415 - heavy, and not needed by the tests of the logic above

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=2)
    )

    def embed(x: np.ndarray) -> np.ndarray:
        stream = extractor.create_stream()
        stream.accept_waveform(RATE, x)
        stream.input_finished()
        return np.array(extractor.compute(stream), dtype=np.float32)

    return embed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        options = json.loads((DATA / "options.json").read_text())
    except (OSError, ValueError):
        options = {}
    voices = Voices(DATA, float(options.get("threshold", 0.7)), float(options.get("margin", 0.06)),
                    bool(options.get("learn", True)))
    _LOGGER.info("Voice ID ready: %d voices, threshold %.2f, margin %.2f", len(voices.people),
                 voices.threshold, voices.margin)
    meaning = None
    wanted = str(options.get("text_model", "base"))
    folders = [Path(TEXT_MODEL) / wanted, Path(TEXT_MODEL) / ("small" if wanted == "base" else "base"), Path(TEXT_MODEL)]
    folder = next((f for f in folders if (f / "model.onnx").exists()), None)
    if folder is not None:
        try:
            model_id = (folder / "MODEL_ID").read_text().strip()
        except OSError:
            model_id = "multilingual-e5"
        meaning = Meaning(TextModel(str(folder)), model_id)
        _LOGGER.info("Text model: %s", model_id)
    else:
        _LOGGER.warning("No text model in %s: semantic memory is off", TEXT_MODEL)
    faces = None
    detector, recognizer = Path(FACE_MODELS) / "yunet.onnx", Path(FACE_MODELS) / "sface.onnx"
    if detector.exists() and recognizer.exists():
        gallery = Voices(DATA, float(options.get("face_threshold", 0.40)), float(options.get("face_margin", 0.05)),
                         bool(options.get("learn", True)), filename="faces.json")
        faces = Faces(gallery, FaceModel(str(detector), str(recognizer)))
        _LOGGER.info("Face recognition ready: %d faces, threshold %.2f", len(gallery.people), gallery.threshold)
    else:
        _LOGGER.warning("No face models in %s: face recognition is off", FACE_MODELS)
    web.run_app(Service(voices, make_embedder(MODEL), meaning, faces).app(),
                port=int(os.environ.get("VOICE_ID_PORT", "8765")), print=None)


if __name__ == "__main__":
    main()
