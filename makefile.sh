# Push current image
docker build -t ${GCP_TRAIN_IMAGE}
docker push ${GCP_TRAIN_IMAGE}