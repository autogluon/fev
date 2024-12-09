from multiprocessing import cpu_count

import datasets

PREDICTIONS = "predictions"
DEFAULT_NUM_PROC = cpu_count()

TRAIN = datasets.Split.TRAIN
FUTURE = datasets.splits.NamedSplit("future")
TEST = datasets.Split.TEST
