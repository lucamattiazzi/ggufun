FROM ollama/ollama:latest

ENV OLLAMA_MODELS=/models

COPY snake/snake.gguf /build/
COPY snake/Modelfile /build/snake.Modelfile

COPY wolf/wolf.gguf /build/
COPY wolf/Modelfile /build/wolf.Modelfile

COPY life/life.gguf /build/
COPY life/Modelfile /build/life.Modelfile

COPY base64/base64.gguf /build/
COPY base64/Modelfile /build/base64.Modelfile

# COPY doom/doom.gguf /build/
# COPY doom/Modelfile /build/doom.Modelfile

COPY kv/kv.gguf /build/
COPY kv/Modelfile /build/kv.Modelfile

COPY pythonvm/pythonvm.gguf /build/
COPY pythonvm/Modelfile /build/pythonvm.Modelfile

# non e' una macchina, e' l'uscita di un compilatore: questo GGUF E' il sito.
# Prima del build: `cd www && python make_catalog.py && python compile_site.py
# afterthebubble` (assembla lo specchio, poi scrive gguf e Modelfile insieme)
COPY www/afterthebubble.gguf /build/
COPY www/afterthebubble.Modelfile /build/afterthebubble.Modelfile

RUN ollama serve & \
    until ollama list >/dev/null 2>&1; do sleep 0.5; done; \
    cd /build && \
    ollama create snake -f snake.Modelfile && \
    ollama create wolf -f wolf.Modelfile && \
    ollama create life -f life.Modelfile && \
    ollama create kv -f kv.Modelfile && \
    ollama create base64 -f base64.Modelfile && \
    ollama create pythonvm -f pythonvm.Modelfile && \
    ollama create afterthebubble -f afterthebubble.Modelfile && \
    cd / && rm -rf /build
