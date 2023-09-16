import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/home/huziyou/pythonFile/IDL/Assignments_0/train_in - Copy.csv')

reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean')
embedding = reducer.fit_transform(data)
plt.scatter(embedding[:, 0], embedding[:, 1], cmap='viridis')
plt.colorbar()
plt.show()
plt.savefig('umap.png')
