import sys
import json
import csv

from os import path
from time import time
from datetime import datetime

import numpy as np
from operator import add
from pyspark.sql import SparkSession

from svm import SVM
from load_data import DataLoader
import settings as s


def fit_then_dump(data, learning_rate, lambda_reg, frac, niter=100, spark=None):
    start_time = time()
    model = SVM(learning_rate, lambda_reg, frac, s.dim)
    fit_log = model.fit(data.training_set, data.validation_set, niter, spark)
    end_time = time()

    training_accuracy = model.predict(data.training_set, spark=spark)
    validation_accuracy = model.predict(data.validation_set, spark=spark)
    valdiation_loss = model.loss(data.validation_set, spark=spark)
    test_accuracy = model.predict(data.test_set, spark=spark)
    # Save results in a log
    log = [{'start_time': datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.utcfromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            'running_time': end_time - start_time,
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy,
            'validation_loss': valdiation_loss,
            'test_accuracy': test_accuracy,
            'fit_log': fit_log,
            'weights': model.getW().tolist()
            }]

    logname = f'{datetime.utcfromtimestamp(end_time).strftime("%Y%m%d_%H%M%S")}_{learning_rate}_{lambda_reg}_{frac}_log.json'
    with open(path.join(s.logpath, logname), 'w') as outfile:
        json.dump(log, outfile)

    return training_accuracy, validation_accuracy, valdiation_loss


def grid_search(data, learning_rates, lambdas, batch_fracs, spark):
    values = [(
        'learning_rate',
        'lambda_reg',
        'frac',
        'training_accuracy',
        'validation_accuracy',
        'validation_loss'
    )]
    for learning_rate in learning_rates:
        for lambda_reg in lambdas:
            for frac in batch_fracs:
                training_accuracy, validation_accuracy, valdiation_loss = fit_then_dump(
                    data, learning_rate, lambda_reg, frac, niter=100, spark=spark)
                values.append((learning_rate, lambda_reg, frac,
                               training_accuracy, validation_accuracy, valdiation_loss))

    with open(path.join(s.logpath, datetime.utcfromtimestamp(time()).strftime("%Y%m%d_%H%M%S") + '_grid_search_results.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(values)


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Spark")\
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    data = DataLoader(spark)

    fit_then_dump(data, s.learning_rate, s.lambda_reg, s.batch_frac, 1000, spark)

    # learning_rates = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    # batch_fracs = [0.005, 0.01, 0.02]
    # lambdas = [1e-6, 1e-5, 1e-4, 1e-3]
    # for i in range(4):
    #     grid_search(data, learning_rates, lambdas, batch_fracs, spark)

    spark.stop()
