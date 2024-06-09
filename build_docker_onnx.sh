cp predict_onnx.sh predict.sh
docker build -f DockerfileONNX -t automlfreiburg .
docker-squash -t automlfreiburg:latest automlfreiburg:latest
rm predict.sh
