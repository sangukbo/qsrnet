
## Use the container (with docker ≥ 19.03)

```
cd docker/
# Build:
docker build --build-arg USER_ID=$UID -t qsrnet:v0 .
# Run:
docker run --gpus all -it --privileged --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/dev:/dev" --volume="/home/sanguk/gitrepo/qsr_docker/qsrnet/qsrnet:/home/appuser/qsrnet" --name=qsrnet qsrnet:v0
docker run --gpus all -it --privileged --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/dev:/dev" --name=qsrnet qsrnet:v0
# Start
docker start -i qsrnet
# differen bash
docker exec -it qsrnet bash

# Grant docker access to host X server to show images
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' qsrnet`
```
