# Debian, not Home Assistant's Alpine base: the models' libraries ship for glibc.
FROM docker.io/library/python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 VOICE_ID_DATA=/data VOICE_ID_MODEL=/models/wespeaker.onnx VOICE_ID_TEXT_MODEL=/models/e5

# The versions the voice prints were tested with.
RUN pip install --no-cache-dir "sherpa-onnx==1.13.8" "numpy>=1.26,<3" "aiohttp>=3.9,<4" \
    "onnxruntime>=1.18,<2" "tokenizers>=0.19,<1"

# WeSpeaker ResNet34 (VoxCeleb), about 26 MB: turns a few seconds of speech into a voice print.
ADD https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx /models/wespeaker.onnx

# multilingual-e5-base, quantised (about 280 MB): sentences in Hungarian and 100 other languages to
# numbers for meaning, for Sentient's semantic memory. Falls back to the small model (about 120 MB) if the
# base one can't be downloaded; MODEL_ID records which it is, so Sentient knows when it changes.
# Both sizes are kept; the app's text_model option picks one (base by default).
RUN python3 - <<'EOF'
import urllib.request, pathlib
files = {"model.onnx": "onnx/model_quantized.onnx", "tokenizer.json": "tokenizer.json"}
got = 0
for size, repo in [("base", "Xenova/multilingual-e5-base"), ("small", "Xenova/multilingual-e5-small")]:
    out = pathlib.Path("/models/e5") / size
    out.mkdir(parents=True, exist_ok=True)
    try:
        for name, path in files.items():
            urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/{path}", out / name)
            print("got", name, "from", repo, (out / name).stat().st_size)
        (out / "MODEL_ID").write_text(repo.split("/")[1])
        got += 1
    except Exception as err:
        print("not from", repo, err)
        for f in out.iterdir():
            f.unlink()
if not got:
    raise SystemExit("could not download a text model")
EOF

# Faces: OpenCV's YuNet (finds faces, 0.2 MB) and SFace (tells them apart, 38 MB).
RUN pip install --no-cache-dir "opencv-python-headless>=4.9,<5"
ADD https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx /models/faces/yunet.onnx
ADD https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx /models/faces/sface.onnx

COPY server.py /server.py

CMD ["python3", "/server.py"]
