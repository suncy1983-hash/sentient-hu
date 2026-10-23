ARG BUILD_FROM
FROM ${BUILD_FROM}

ENV PYTHONUNBUFFERED=1 VOICE_ID_DATA=/data VOICE_ID_MODEL=/models/wespeaker.onnx

# The versions the voice prints were tested with.
RUN pip install --no-cache-dir "sherpa-onnx==1.13.8" "numpy>=1.26,<3" "aiohttp>=3.9,<4"

# WeSpeaker ResNet34 (VoxCeleb), about 26 MB: turns a few seconds of speech into a voice print.
ADD https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx /models/wespeaker.onnx

COPY server.py /server.py

CMD ["python3", "/server.py"]
