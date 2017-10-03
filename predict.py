import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

housing = pd.read_csv("london/housing.csv")

housing.hist(bins=50, figsize=(20,15))

plt.savefig("image.png",bbox_inches='tight',dpi=100)

# CREATE TEST SERS

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)

#eliminating outliers sort of
housing["price"].where(housing["price"] < 2000, 2000, inplace=True)

#stratified sampling
