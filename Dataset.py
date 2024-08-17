import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

names = []
base = 'Data/'
with os.scandir(base) as entries:
    for entry in entries:
        if(entry.is_file() == False):
            names.append(entry.name)
print(names)