import numpy as np
import tensorflow as tf

tf.set_random_seed(0)

'''
データの生成
'''
# XORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

'''
モデル設定
'''
# n行2列  [[0, 0], [0, 1], [1, 0], [1, 1]]
x = tf.placeholder(tf.float32, shape=[None, 2])
# n行1列 [[0], [1], [1], [0]]
t = tf.placeholder(tf.float32, shape=[None, 1])

# 入力層 - 隠れ層
# 2行2列の行列 (つまり、入力2で出力2)
W = tf.Variable(tf.truncated_normal([2, 2]))
# 1行2列の行列
b = tf.Variable(tf.zeros([1, 2]))
h = tf.nn.sigmoid(tf.matmul(x, W) + b)

# 隠れ層 - 出力層
# 2行1列の行列 (つまり、入力2で出力1)
V = tf.Variable(tf.truncated_normal([2, 1]))
c = tf.Variable(tf.zeros([1,1]))
y = tf.nn.sigmoid(tf.matmul(h, V) + c)

# 2値の場合のクロスエントロピー誤差
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)

'''
モデル学習
'''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(4000):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })
    if epoch % 1000 == 0:
        print('epoch:', epoch)

'''
学習結果の確認
'''
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
prob = y.eval(session=sess, feed_dict={
    x: X
})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)
