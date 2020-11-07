# Qualitative Spatial Representation Network (QSRNet) Repository

This repository contains implementation of qualitative spatial representation network (QSRNet).

---------------

## How to use

1. Clone the code.

    ```
    git clone git@github.com:sangukbo/qsrnet.git
    ```

2. Build the docker image.

    We used docker with version ≥ 19.03.

    ```
    cd docker/
    docker build --build-arg USER_ID=$UID -t qsrnet:v0 .
    ```

3. Run the docker image.

    ```
    docker run --gpus all -it --privileged --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/dev:/dev" --name=qsrnet qsrnet:v0
    ```

4. Start the docker image and run ``setup.py``

    ```
    docker start -i qsrnet
    ```

    Then, in the docker bash shell, run ``setup.py``.

    ```
    python setup.py install --user
    ```

5. Execute another shell with the QSRNet docker image

    You will need to run two docker bash shells for QSRNet. One is for the neural network part which computes the metrics and the other is for the dynamic Bayesian network (DBN) part which computes the qualitative spatial representations. The two parts communicate with each other using python socket.

    For this, open a new terminal and execute another docker bash shell with QSRNet image.

    ```
    docker exec -it qsrnet bash
    ```

5. Run the QSRNet.

    First, you run ``qsrnet_dbn.py`` on one of the docker bash shell.

    ```
    python qsrnet_dbn.py
    ```

    Next, run ``python qsrnet_compute_metrics.py`` on the other docker bash shell.

    ```
    python qsrnet_compute_metrics.py
    ```

---------------

## Acknowledgements

1. The ``dockerfile`` for the QSRNet is based on that for the [detectron2](https://github.com/facebookresearch/detectron2).

2. The python project ``setup.py`` file for the QSRNet is based on these pages ([simple python project](http://www.kennethreitz.org/essays/repository-structure-and-python), [setup.py file](http://www.kennethreitz.org/essays/repository-structure-and-python)).
