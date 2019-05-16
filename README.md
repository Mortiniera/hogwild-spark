# Hogwild spark

## Authors 

- Timoté Vaucher
- Patrik Wagner
- Thevie Mortiniera

## Setup
We suppose one have a version of [Spark including Hadoop](https://spark.apache.org/downloads.html) , [Anaconda](https://www.anaconda.com/distribution/), Docker, Kubernetes (i.e. `kubectl`) installed on your computer

Set up the environment
```bash
conda create -n hogwild-spark python=3.7 scipy numpy pyspark
source activate hogwild-spark
```

## Build the image

You first need to set the env var `SPARK_HOME` and update the spark_home arg in the `Dockerfile` in order to be able to build the image. If you want to change the name of the image, this can be done in `docker-image-tool.sh`

Then to build and push the image :

```bash
bash docker-image-tool.sh -t tag build
bash docker-image-tool.sh -t tag push
```

*note : `numpy` and `scipy` take a long time to build the first time you create the image* 

## Run the app

You first need to update the `GROUP_NAME` and `IMAGE` in `run.sh`

Then to run the app (`-w` for the number of executor)

```bash
bash run.sh -w 4
```

Alternatively one can combine the 2 steps (build / call) by using the additional args

```bash
bash run.sh -w 4 -t tag -n AppName -b
```

## Get the logs

The simplest way to access the logs stored on the container is to create another pod and then use `kubectl cp`

```bash
kubectl create -f helper/shell.yml
kubectl cp shell-pod:/data/logs path/to/your/logs
```

### Access a bash

To see a bash and inspect the `\data` folder on can use the above pod by calling `kubectl attach -t -i shell-pod` (this also allows you to delete old logs if necessary)
