# Push current image
source devops/.env
docker buildx build -t ${IMAGE_NAME}:${IMAGE_TAG} --file devops/Dockerfile . --platform=linux/amd64
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${GCP_TRAIN_IMAGE}
docker push ${GCP_TRAIN_IMAGE}