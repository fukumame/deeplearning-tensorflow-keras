import numpy as np
import tensorflow as tf
tf.set_random_seed(0)  # 乱数シード

'''
wは2行1列
[[ 0.]
 [ 0.]]
'''
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

'''
Noneはn行のこと (可変長配列)
つまり、n行2列の配列
'''
x = tf.placeholder(tf.float32, shape=[None, 2])
# tは正解用のラベルを格納するための配列
t = tf.placeholder(tf.float32, shape=[None, 1])

# 行列の掛け算として、x * w　をかけている。 (逆だと行列のサイズが合わずにエラーになる)
y = tf.nn.sigmoid(tf.matmul(x, w) + b)

# 交差エントロピー誤差 (この値を最小化することを目的とする)
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
# 0.1は学習率
# 交差エントロピー誤差を最小化するための関数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# モデルの出力結果が正解ラベルとあっているかどうかを表す関数 (あっていた場合True, 間違っていた場合False)
# tf.greater(y, 0.5) は y> 0.5の場合はTrueが返り、y<=0.5の場合はFalseが返る
# tf.to_float(tf.greater(y, 0.5)) とすることで、bool値をfloat値に変換する。
# つまり、Trueの場合は 1.0、Falseの場合は0が返るようになる
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

'''
モデル学習
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 正解ラベル
Y = np.array([[0], [1], [1], [1]])

# 変数の初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 変数の値を見たい場合はこのようにする
print(sess.run(w))

# 学習
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

'''
学習結果の確認
'''
# モデルの出力結果があっていたかどうか？
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})

# 予測モデルの出力結果そのもの
prob = y.eval(session=sess, feed_dict={
    x: X
})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)
