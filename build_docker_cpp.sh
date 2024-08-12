cp predict_cpp.sh predict.sh
docker build -f DockerfileCpp -t automlfreiburg .
docker-squash -t automlfreiburg:latest automlfreiburg:latest
rm predict.sh
