# 모델 인 이미지 패턴

## 목적
학습으로 생성된 모델 파일을 포함한 추론 서버 이미지 빌드

## 사용법
1. 추론용 Docker image 빌드
```make build_all```

2. 추론 서버를 k8s에 배포
```make deploy```

3. k8s에서 추론 서버 삭제
```make delete```