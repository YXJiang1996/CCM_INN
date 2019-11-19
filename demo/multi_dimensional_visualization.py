import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
cols = ['Class', 'Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols',
        'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity',
        'Hue', 'OD280/OD315', 'Proline']
data = pd.read_csv(url, names=cols)

y = data['Class']  # Split off classifications
X = data.ix[:, 'Alcohol':]  # Split off features
X_norm = (X - X.min()) / (X.max() - X.min())  # feature scaling

"""
# Method 1: Two-dimensional slices
# three different scatter series so the class labels in the legend are distinct
plt.scatter(X[y==1]['Flavanoids'], X[y==1]['NonflavanoidPhenols'], label='Class 1', c='red')
plt.scatter(X[y==2]['Flavanoids'], X[y==2]['NonflavanoidPhenols'], label='Class 2', c='blue')
plt.scatter(X[y==3]['Flavanoids'], X[y==3]['NonflavanoidPhenols'], label='Class 3', c='lightgreen')

plt.legend()
plt.xlabel('Flavanoids')
plt.ylabel('NonflavanoidPhenols')
plt.show()
"""

"""
# Method 2: PCA Plotting
pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))
plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label='Class 1', c='red')
plt.scatter(transformed[y == 2][0], transformed[y == 2][1], label='Class 2', c='blue')
plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Class 3', c='lightgreen')

plt.legend()
plt.show()
"""

"""
# Method 3: Linear Discriminant Analysis
lda = LDA(n_components=2)  # 2-dimensional LDA
lda_transformed = pd.DataFrame(lda.fit_transform(X_norm, y))
plt.scatter(lda_transformed[y == 1][0], lda_transformed[y == 1][1], label='Class 1', c='red')
plt.scatter(lda_transformed[y == 2][0], lda_transformed[y == 2][1], label='Class 2', c='blue')
plt.scatter(lda_transformed[y == 3][0], lda_transformed[y == 3][1], label='Class 3', c='lightgreen')

plt.legend(loc=3)
plt.show()
"""
'''
# Method 4: Parallel Coordinates
# Select features to include in the plot
plot_feat = ['MalicAcid', 'Ash', 'OD280/OD315', 'Magnesium','TotalPhenols']

# Concat classes with the normalized data
data_norm = pd.concat([X_norm[plot_feat], y], axis=1)

# Perform parallel coordinate plot
parallel_coordinates(data_norm, 'Class')
plt.show()
'''

# Method 5: TSNE
tsne = TSNE(n_components=2, init='pca')
transformed = pd.DataFrame(tsne.fit_transform(X_norm))
plt.scatter(transformed[y == 1][0], transformed[y == 1][1], label='Class 1', c='red')
plt.scatter(transformed[y == 2][0], transformed[y == 2][1], label='Class 2', c='blue')
plt.scatter(transformed[y == 3][0], transformed[y == 3][1], label='Class 3', c='lightgreen')

plt.legend()
plt.show()
