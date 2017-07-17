import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# Wineデータセットの読み込み#
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

# 列名を設定 #
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
df_wine.head()

# 特徴量とクラスラベルを別々に抽出 #
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

#　トレーニングデータ70％とテストデータ30％に分割　#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 最小値と最大値のスケーリング #
#mms = MinMaxScaler()
#X_train_norm = mms.fit_transform(X_train)
#X_test_norm = mms.transform(X_test)

# 標準化のインスタンスを作成 #
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 共分散行列を作成 #
cov_mat = np.cov(X_train_std.T)

# 固有値と固有ベクトルを計算 #
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# P128 129 #
# 固有値を合計 #
tot = sum(eigen_vals)
# 分散説明率を計算 #
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
#　分散説明率の累積和を取得　#
cum_var_exp = np.cumsum(var_exp)
# 分散説明率の棒グラフを作成 #
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
# 分散説明率の累積和の階段グラフを作成  #
plt.step(range(1, 14), cum_var_exp, where='mid',
                  label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('./figures/pca1.png', dpi=300)
plt.show()

# 特徴変換 #
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

# Note: I added the `key=lambda k: k[0]` in the sort call above
# just like I used it further below in the LDA section.
# This is to avoid problems if there are ties in the eigenvalue
# arrays (i.e., the sorting algorithm will only regard the
# first element of the tuples, now).
print("==================================================================================")
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))#,
#               eigen_pairs[2][1][:, np.newaxis],
#               eigen_pairs[3][1][:, np.newaxis],
#               eigen_pairs[4][1][:, np.newaxis],
#               eigen_pairs[5][1][:, np.newaxis],
#               eigen_pairs[6][1][:, np.newaxis],
#               eigen_pairs[7][1][:, np.newaxis],
#               eigen_pairs[8][1][:, np.newaxis],
#               eigen_pairs[9][1][:, np.newaxis],
#               eigen_pairs[10][1][:, np.newaxis],
#               eigen_pairs[11][1][:, np.newaxis],
#               eigen_pairs[12][1][:, np.newaxis]))

# 124*13次元のトレーニングデータセット全体をw次元に変換して出力#
#print(X_train_std.dot(w))

#print('Matrix W:\n', w)
#print(X_train_std[0].dot(w))
X_train_pca = X_train_std.dot(w)
X_test_pca = X_test_std.dot(w)
#colors = ['r', 'b', 'g']
#markers = ['s', 'x', 'o']

#for l, c, m in zip(np.unique(y_train), colors, markers):
#        plt.scatter(X_train_pca[y_train == l, 0],
#                                    X_train_pca[y_train == l, 1],
#                                    c=c, label=l, marker=m)

#        plt.xlabel('PC 1')
#        plt.ylabel('PC 2')
#        plt.legend(loc='lower left')
#        plt.tight_layout()
        # plt.savefig('./figures/pca2.png', dpi=300)
#plt.show()


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

            # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
                plt.scatter(x=X[y == cl, 0],
                            y=X[y == cl, 1],
                            alpha=0.6,
                            c=cmap(idx),
                            edgecolor='black',
                            marker=markers[idx],
                            label=cl)


lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
print(lr.score(X_train_pca,y_train))
print(lr.score(X_test_pca,y_test))

#plot_decision_regions(X_train_pca, y_train, classifier=lr)
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.legend(loc='lower left')
#plt.tight_layout()
# plt.savefig('./figures/pca3.png', dpi=300)
#plt.show()
