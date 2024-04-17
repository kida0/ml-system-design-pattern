FROM python:3.8-slim

ENV PROJECT_DIR /app
WORKDIR $PROJECT_DIR

COPY ./requiremets.txt ./

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir -r requiremets.txt && \
    apt-get remove -y gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY ./src/ ./src/
COPY ./models/ ./models/

ENV MODEL_FILEPATH $PROJECT_DIR/models/iris_svc.onnx
ENV LABEL_FILE_PATH $PROJECT_DIR/models/label.json
ENV LOG_LEVEL DEBUG
ENV LOG_FORMAT TEXT

COPY ./run.sh ./
RUN chmod +x ./run.sh
CMD [ "/run.sh" ]