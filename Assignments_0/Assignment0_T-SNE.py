import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
data = pd.read_csv('/home/huziyou/pythonFile/IDL/Assignments_0/train_in - Copy.csv')
embedding = tsne.fit_transform(data)
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.xlabel('T-SNE Dimension 1')
plt.ylabel('T-SNE Dimension 2')
plt.title('T-SNE Visualization')
plt.show()
plt.savefig('T-SNE.png')
