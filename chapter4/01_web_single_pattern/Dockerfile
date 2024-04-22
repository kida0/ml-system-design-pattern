FROM python:3.8-slim AS base

ENV PROJECT_DIR /app
WORKDIR ${PROJECT_DIR}

COPY ./requirements.txt .

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends apt-utils gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y apt-utils gcc && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    
FROM base

COPY ./src/ ./src/
COPY ./models/ ./models/

ENV MODEL_FILEPATH ./models/iris_svc.onnx
ENV LABEL_FILEPATH ./models/labels.json
ENV LOG_LEVEL DEBUG
ENV LOG_FORMAT TEXT

COPY ./run.sh .
RUN chmod +x ./run.sh

CMD [ "./run.sh" ]