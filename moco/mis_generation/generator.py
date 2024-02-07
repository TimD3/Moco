from abc import ABC, abstractmethod, abstractstaticmethod
from pathlib import Path
import os
import shutil
import pickle
import json
import numpy as np
import multiprocessing


class DataGenerator(ABC):

    def random_weight(self, n, mu = 1, sigma = 0.1):
        return np.around(np.random.normal(mu, sigma, n)).astype(int).clip(min=0)

    @abstractmethod
    def generate(self, gen_labels = False, weighted = False):
        pass
