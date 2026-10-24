# Debian, not Home Assistant's Alpine base: the models' libraries ship for glibc.
FROM docker.io/library/python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 VOICE_ID_DATA=/data VOICE_ID_MODEL=/models/wespeaker.onnx VOICE_ID_TEXT_MODEL=/models/e5

# The versions the voice prints were tested with.
RUN pip install --no-cache-dir "sherpa-onnx==1.13.8" "numpy>=1.26,<3" "aiohttp>=3.9,<4" \
    "onnxruntime>=1.18,<2" "tokenizers>=0.19,<1"

# WeSpeaker ResNet34 (VoxCeleb), about 26 MB: turns a few seconds of speech into a voice print.
ADD https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx /models/wespeaker.onnx

# multilingual-e5-small, quantised (about 120 MB): sentences in Hungarian and 100 other languages to
# numbers for meaning, for Sentient's semantic memory. Tries a second copy if the first is missing.
RUN python3 - <<'EOF'
import urllib.request, pathlib
out = pathlib.Path("/models/e5"); out.mkdir(parents=True, exist_ok=True)
repos = ["Xenova/multilingual-e5-small", "MintplexLabs/multilingual-e5-small"]
files = {"model.onnx": "onnx/model_quantized.onnx", "tokenizer.json": "tokenizer.json"}
for name, path in files.items():
    for repo in repos:
        try:
            urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/{path}", out / name)
            print("got", name, "from", repo, (out / name).stat().st_size)
            break
        except Exception as err:
            print("not from", repo, err)
    else:
        raise SystemExit(f"could not download {name}")
EOF

COPY server.py /server.py

CMD ["python3", "/server.py"]
