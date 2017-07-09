import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(0)

M = 2      # 入力データの次元
K = 3      # クラス数
n = 100    # クラスごとのデータ数
N = n * K  # 全データ数

'''
データの生成
'''
X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

# X1, X2, S3の結合
X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)

'''
モデル設定
'''
# 2行3列の行列 [[0, 0, 0], [0, 0, 0]]
W = tf.Variable(tf.zeros([M, K]))
# 1行3列の配列 [[0, 0, 0]]
b = tf.Variable(tf.zeros([1, K]))

# n行2列の行列
# [[1,3], [3,6], ...] といった配列を想定
x = tf.placeholder(tf.float32, shape=[None, M])

# n行3列の行列
# [[1,0,1], [0,0,1], ...]といった配列を想定
t = tf.placeholder(tf.float32, shape=[None, K])

# ソフトマックス関数の定義
y = tf.nn.softmax(tf.matmul(x, W) + b)

'''
reduction_indicesは[1]を指定すると、例えば [[1,2,3], [4,5,6]]
の場合、1+2+3 = 6と4+5+6=15を行い、[6, 15]が返る
なお、このオプションは axisに置き換えられる予定。
axis = 1を指定すると同じ結果となる。
axis = 0を指定すると、[5,7,9]が返る
詳しくは以下を参照のこと
https://www.tensorflow.org/api_docs/python/tf/reduce_sum
'''
cross_entropy_sum = -tf.reduce_sum(t * tf.log(y), axis=1)
cross_entropy = tf.reduce_mean(cross_entropy_sum)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

'''
argmaxは与えられた行列(リスト)の中での最大値である配列の位置を返す
第2引数は最大値を計算する軸のこと
1を指定すると、例えば、[[1,2,3], [4,5,6]]の場合、[2, 2]を返す (それぞれ、3と6が最大であるため)
一方、0を指定すると[4, 5, 6]を返す。
'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))

'''
モデル学習
'''
# 変数の初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 50  # ミニバッチサイズ
# 全データ数 = 300を50で割った数、すなわち 6
n_batches = N // batch_size

# ミニバッチ学習
for epoch in range(20):
    # XとYをシャッフル
    X_, Y_ = shuffle(X, Y)

    # iは0~5の値を取りうる
    for i in range(n_batches):

        '''
        Start, Endはそれぞれ、以下の値を取りうる
        [0, 50], [50, 100], [100, 150] ...
        つまりデータを50ずつ分割して学習させている
        '''
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })

'''
学習結果の確認
'''
X_, Y_ = shuffle(X, Y)

classified = correct_prediction.eval(session=sess, feed_dict={
    x: X_[0:10],
    t: Y_[0:10]
})
prob = y.eval(session=sess, feed_dict={
    x: X_[0:10]
})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)

print(sess.run(tf.zeros([1, K])))
