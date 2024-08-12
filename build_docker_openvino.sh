cp predict_openvino.sh predict.sh
docker build -f DockerfileOpenVINO -t automlfreiburg .
docker-squash -t automlfreiburg:latest automlfreiburg:latest
rm predict.sh
