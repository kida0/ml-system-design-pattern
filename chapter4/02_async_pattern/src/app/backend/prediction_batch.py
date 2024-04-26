import asyncio
import base64
import io
import os
from concurrent.futures import ProcessPoolExecutor
from logging import DEBUG, Formatter, StreamHandler, getLogger
from time import sleep

import grpc
from src.app.backend import request_inception_v3, store_data_job
from src.configurations import CacheConfigurations, ModelConfigurations
from tensorflow_serving.apis import prediction_service_pb2_grpc

log_format = Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")
logger = getLogger("prediction_batch")
stdout_handler = StreamHandler()
stdout_handler.setFormatter(log_format)
logger.addHandler(stdout_handler)
logger.setLevel(DEBUG)


def _trigger_prediction_if_queue(stub: prediction_service_pb2_grpc.PredictionServiceStub):
    """ 
    1. redis 큐에서 작업 id를 가져와서 tf-serving grpc로 이미지 분류 요청
    2. 예측 결과를 redis에 저장하거나 실패한 경우 작업을 다시 큐에 push
    """
    job_id = store_data_job.right_pop_queue(CacheConfigurations.queue_name)
    logger.info(f"predict job_id: {job_id}")
    if job_id is not None:
        data = store_data_job.get_data_redis(job_id)
        if data != "":
            return True
        image_key = store_data_job.make_image_key(job_id)
        image_data = store_data_job.get_data_redis(image_key)
        decoded = base64.b64decode(image_data)
        io_bytes = io.BytesIO(decoded)
        prediction = request_inception_v3.request_grpc(
            stub=stub,
            image=io_bytes.read(),
            model_spec_name=ModelConfigurations.model_spec_name,
            signature_name=ModelConfigurations.signature_name,
            timeout_second=5,
        )
        if prediction is not None:
            logger.info(f"{job_id} {prediction}")
            store_data_job.set_data_redis(job_id, prediction)
        else:
            store_data_job.left_push_queue(CacheConfigurations.queue_name, job_id)


def _loop():
    """ 루프를 실행하면서 1초마다 위 함수 호출하여 예측 작업 수행"""
    serving_address = f"{ModelConfigurations.address}:{ModelConfigurations.grpc_port}"
    channel = grpc.insecure_channel(serving_address)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    while True:
        sleep(1)
        _trigger_prediction_if_queue(stub=stub)


def prediction_loop(num_procs: int = 2):
    """ 
    1. 프로세스 풀 실행기 생성 -> 비동기 이벤트 루프를 가져와서 지정된 프로세스 수 만큼 _loop 함수 실행 
    2. 이벤트 루프를 시작하여 예측 작업을 병렬로 처리
    """
    executor = ProcessPoolExecutor(num_procs)
    loop = asyncio.get_event_loop()

    for _ in range(num_procs):
        asyncio.ensure_future(loop.run_in_executor(executor, _loop))

    loop.run_forever()


def main():
    """ 환경 변수에서 프로세스 수를 가져오거나 기본갑승로 2 사용하여 prediction_loop 수행 """
    NUM_PROCS = int(os.getenv("NUM_PROCS", 2))
    prediction_loop(NUM_PROCS)


if __name__ == "__main__":
    logger.info("start backend")
    main()