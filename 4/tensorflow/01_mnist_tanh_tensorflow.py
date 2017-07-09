import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(123)

'''
データの生成
'''
mnist = datasets.fetch_mldata('MNIST original', data_home='.')

# n = 70000
n = len(mnist.data)
N = 10000  # MNISTの一部を使う
train_size = 0.8

# np.random.permutation(range(n))によって、70000までの数字がランダムに配列に格納される
# 例えば、array([19248, 22556, 35706, ..., 13364,  3970,  7771])となる
# [:N]によって、先問から 10000個までが選択される。
# 結果として1万個分の要素のINDEXがランダムに取得され、70000要素のうちの10000個がランダムに取得できる
indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

# mnist.dataによって、元の画像データが取得できる
# 元の画像データは、784の要素を持つ、1次元配列。(要素の取りうる値は 0~255であり、色の濃さを表す)
# indicesによって、指定された画像データがランダムに取得できる
'''
array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       128, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0, 128, 255, 191,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0, 255, 255, 128,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,  64, 255, 255,  64,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0, 191, 255, 191,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0, 128, 255, 255, 128,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 191, 255,
       191,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 191,
       255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0, 255, 255, 191,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0, 128, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,  64, 255, 255, 128,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0, 191, 255, 255,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,  64, 255, 255,  64,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0, 191, 255, 255,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 128,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 128, 255,
       191,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       191, 255, 128,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0, 128, 255, 255, 128,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,  64, 255, 255, 128,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0], dtype=uint8)
'''
X = mnist.data[indices]
# mnist.targetは正解ラベルデータ
# array([ 1.,  9.,  2., ...,  6.,  3.,  6.])
y = mnist.target[indices]


'''
numpy.eye(10)によって、以下のような10x10の単位行列が生成できる。
array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
y.astype(int)によって、yをint型に変換している
つまり、yが2の場合、[ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]がYに入る
'''
Y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

# 学習用教師データと検証用データに分ける
X_train, X_test, Y_train, Y_test =\
    train_test_split(X, Y, train_size=train_size)

'''
モデル設定
'''
n_in = len(X[0])  # 784
n_hidden = 200
n_out = len(Y[0])  # 10

# 入力データ (画像データ) の入れ物
x = tf.placeholder(tf.float32, shape=[None, n_in])
# 出力データ (正解ラベル) の入れ物
t = tf.placeholder(tf.float32, shape=[None, n_out])

# 入力層 - 隠れ層
'''
tf.truncated_normalはTensorを正規分布かつ標準偏差の２倍までのランダムな値で初期化するための関数
http://qiita.com/supersaiakujin/items/464cc053418e9a37fa7b#truncated_normal
'''
# 784行 200列の配列
W0 = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.1))
# 1行 200列の配列
b0 = tf.Variable(tf.zeros([1, n_hidden]))
# h0はn行200列の配列になる(nはxの要素数)、つまり、学習用データの画像データ数
# 3つの画像データが含まれていた場合 n = 3となる
# 活性化関数としてtanhを使っている
h0 = tf.nn.tanh(tf.matmul(x, W0) + b0)

# 隠れ層 - 隠れ層
W1 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.1))
b1 = tf.Variable(tf.zeros([1, n_hidden]))
h1 = tf.nn.tanh(tf.matmul(h0, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.1))
b2 = tf.Variable(tf.zeros([1, n_hidden]))
h2 = tf.nn.tanh(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.1))
b3 = tf.Variable(tf.zeros([1, n_hidden]))
h3 = tf.nn.tanh(tf.matmul(h2, W3) + b3)

# 隠れ層 - 出力層
W4 = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.1))
b4 = tf.Variable(tf.zeros([1, n_out]))
y = tf.nn.softmax(tf.matmul(h3, W4) + b4)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y),
                               reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
モデル学習
'''
epochs = 100
## ミニバッチの大きさ
batch_size = 200

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# ミニバッチを用いた学習の繰り返し回数
n_batches = (int)(N * train_size) // batch_size

for epoch in range(epochs):
    X_, Y_ = shuffle(X_train, Y_train)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end]
        })

    # 訓練データに対する学習の進み具合を出力
    loss = cross_entropy.eval(session=sess, feed_dict={
        x: X_,
        t: Y_
    })
    acc = accuracy.eval(session=sess, feed_dict={
        x: X_,
        t: Y_
    })
    print('epoch:', epoch, ' loss:', loss, ' accuracy:', acc)

'''
予測精度の評価
'''
accuracy_rate = accuracy.eval(session=sess, feed_dict={
    x: X_test,
    t: Y_test
})
print('accuracy: ', accuracy_rate)
