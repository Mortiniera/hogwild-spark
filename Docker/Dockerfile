FROM openjdk:8-alpine

ARG spark_folder=spark-2.4.1-bin-hadoop2.7
ARG spark_jars=jars

RUN set -ex && \
    apk upgrade --no-cache && \
    apk add --no-cache bash tini libc6-compat linux-pam && \
    mkdir -p /opt/spark && \
    mkdir -p /opt/spark/work-dir && \
    touch /opt/spark/RELEASE && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd

RUN mkdir ${SPARK_HOME}/python
RUN apk add --no-cache python3 && \
    python3 -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    pip3 install --upgrade pip setuptools && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip && \
    rm -r /root/.cache

RUN apk update && apk add --no-cache \
    libstdc++ \
    libgomp \
    build-base \
    cmake \
    gfortran \
    libpng && \
    ln -s locale.h /usr/include/xlocale.h && \
    apk add --no-cache --virtual .build-deps \
    lapack-dev \
    musl-dev \
    python3-dev \
    jpeg-dev \
    freetype-dev \
    libffi-dev \
    openssl-dev \
    g++

COPY requirements.txt /opt
RUN pip install --no-cache-dir -r /opt/requirements.txt

COPY ${spark_folder}/${spark_jars} /opt/spark/jars
COPY ${spark_folder}/bin /opt/spark/bin
COPY ${spark_folder}/sbin /opt/spark/sbin
COPY Docker/entrypoint.sh /opt

ENV SPARK_HOME /opt/spark
COPY ${spark_folder}/python/lib ${SPARK_HOME}/python/lib
ENV PYTHONPATH ${SPARK_HOME}/python/lib/pyspark.zip:${SPARK_HOME}/python/lib/py4j-*.zip

RUN mkdir -p /opt/spark/work-dir/logs
COPY src /opt/spark/work-dir

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]
