FROM ollama/ollama:latest

ENV OLLAMA_MODELS=/models

COPY snake/snake.gguf /build/
COPY snake/Modelfile /build/snake.Modelfile
COPY wolf/wolf.gguf /build/
COPY wolf/Modelfile /build/wolf.Modelfile

RUN ollama serve & \
    until ollama list >/dev/null 2>&1; do sleep 0.5; done; \
    cd /build && ollama create snake -f snake.Modelfile && ollama create wolf -f wolf.Modelfile && cd / && rm -rf /build
