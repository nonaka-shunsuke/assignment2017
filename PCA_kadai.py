import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
#plt.show()

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


train_score=[]
test_score=[]

for i in range(len(eigen_pairs)):
        if i ==0:
                w = eigen_pairs[0][1][:, np.newaxis]
        else:
                w = np.hstack((w, eigen_pairs[i][1][:, np.newaxis]))

        X_train_pca = X_train_std.dot(w)
        X_test_pca = X_test_std.dot(w)

        lr = LogisticRegression()
        lr = lr.fit(X_train_pca, y_train)
        train_score.append(lr.score(X_train_pca,y_train))
        test_score.append(lr.score(X_test_pca,y_test))

pca = PCA(n_components=None)                            
X_train_pca = pca.fit_transform(X_train_std)            
print(pca.explained_variance_ratio_)

_sum=0
print("==================================================================================")
for i in range(len(train_score)):
        _sum+=pca.explained_variance_ratio_[i]
        print("train score:",train_score[i]," test score:{0:.12f}".format(test_score[i]), "分散説明率:{0:.12f}".format(pca.explained_variance_ratio_[i]) ,"分散説明率の累計:", _sum)
