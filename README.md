# Qualitative Spatial Representation Network (QSRNet) Repository

This repository contains implementation of qualitative spatial representation network (QSRNet).

---------------

## Prerequisites

1. A computer with GPU and docker (docker version ≥ 19.03) installed.

2. A webcam

    We used RealSense D435i. If you want to use a different webcam, you might have to modify ``realsense_camera`` class in ``qsrnet/utils/camera_util.py`` for the webcam you have.

---------------

## How to use

1. Clone the code.

    ```
    git clone git@github.com:sangukbo/qsrnet.git
    ```

2. Build the docker image.

    ```
    cd qsrnet/docker/
    docker build --build-arg USER_ID=$UID -t qsrnet:v0 .
    ```

3. Run the docker container.

    First, connect a webcam to your computer.

    ```
    docker run --gpus all -it --privileged --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/dev:/dev" --name=qsrnet qsrnet:v0
    ```

    We assume your webcam is located in ```/dev``` directory. If your webcam is located in a different directory, you cmight have to modify ```--volume="/dev:/dev"```. Here, it should be ```--volume="[directory for your webcam]:/dev"```.

    To start an existing docker container, ``docker start`` insread.

    ```
    docker start -i qsrnet
    ```

4. Execute another shell with the QSRNet docker container

    You will need to run two docker bash shells for QSRNet. One is for the neural network part which computes the metrics and the other is for the dynamic Bayesian network (DBN) part which computes the qualitative spatial representations. The two parts communicate with each other using python socket.

    For this, open a new terminal and execute another docker bash shell with the QSRNet docker container.

    ```
    docker exec -it qsrnet bash
    ```

5. Run the QSRNet.

    First, you run ``qsrnet_dbn.py`` on one of the docker bash shell.

    ```
    cd qsrnet/
    python qsrnet_dbn.py
    ```

    Next, run ``python qsrnet_compute_metrics.py`` on the other docker bash shell.

    ```
    cd qsrnet/
    python qsrnet_compute_metrics.py
    ```

---------------

## Notes

1. If you want to change the transition probability and the observation model for the DBN part, please modify ``qsrnet/dbn/conditional_probabilities.py``. In fact, if you are not getting satisfactory estimation results, you would have to tune the probabilities in this file.

---------------

## Acknowledgements

1. The ``dockerfile`` for the QSRNet is based on that for the [detectron2](https://github.com/facebookresearch/detectron2).

2. The python project ``setup.py`` file for the QSRNet is based on these pages ([simple python project](http://www.kennethreitz.org/essays/repository-structure-and-python), [setup.py file](https://github.com/kennethreitz/setup.py)).
