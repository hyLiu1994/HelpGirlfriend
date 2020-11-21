import tensorflow as tf

# 练习张量加减乘除
A = tf.Variable([[[1, 2], [2, 3], [3, 4]],
                 [[4, 5], [5, 6], [6, 7]],
                 [[7, 8], [8, 9], [9, 10]]])
B = tf.Variable([[[1, 2],
                  [3, 4],
                  [5, 6]]])
# print(A)
# print(B)
# print(A*B)
# print(A-B)
# print(A+B)
# print(A/B)

# 练习张量矩阵运算
A = tf.Variable([[[1, 2], [2, 3], [3, 4]],
                 [[4, 5], [5, 6], [6, 7]],
                 [[7, 8], [8, 9], [9, 10]]])
B = tf.ones(shape=(1, 2, 3), dtype=tf.int32)

C = tf.linalg.matmul(A, B)
# print("A = ", A)
# print("B = ", B)
# print("C = ", C)

# tf.transpose 练习
A = tf.Variable([[[1, 2, 3],
                 [4, 5, 6]]])
B = tf.transpose(A, perm=[2, 1, 0])
# print("A = ", A)
# print("B = ", B)

# tf.reshape 练习
A = tf.Variable([[[1, 2, 3],
                 [4, 5, 6]]])
B = tf.reshape(A, shape=(2, 3))
# print("A = ", A)
# print("B = ", B)

# tf.expand_dims 练习
A = tf.Variable([[[1, 2, 3],
                 [4, 5, 6]]])
B = tf.expand_dims(A, axis=2)
# print("A = ", A)
# print("B = ", B)

# tf.squeeze 练习
A = tf.Variable([[[1, 2, 3],
                 [4, 5, 6]]])
B = tf.squeeze(A, axis=0)
# print("A = ", A)
# print("B = ", B)

# tf.tile 练习
A = tf.Variable([[[1, 2, 3],
                 [4, 5, 6]]])
B = tf.tile(A, multiples=[2, 1, 1])
C = tf.tile(A, multiples=[2, 1, 2])
# print("A = ", A)
# print("B = ", B)
# print("C = ", C)

# tf.cast 练习
A = tf.Variable([[[1, 2, 3],
                 [4, 5, 6]]])
B = tf.cast(A, dtype=tf.float32)
# print("A = ", A)
# print("B = ", B)

# tf.reduce_sum 练习
A = tf.Variable([[[1, 2, 3],
                 [4, 5, 6]]])
B = tf.reduce_sum(A, axis=0)
C = tf.reduce_sum(A)
# print("A = ", A)
# print("B = ", B)
# print("C = ", C)

# 例1
X = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=(32, 50, 1)))
W = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=(60, 50)))
b = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=(60, 1)))

Y = tf.linalg.matmul(tf.expand_dims(W, axis=0), X) + tf.expand_dims(b, axis=0)
Y = tf.keras.activations.sigmoid(Y)
# print("X.shape = ", X.shape, "W.shape = ",
#      W.shape, "b.shape = ", b.shape, "Y.shape = ", Y.shape)

# 例2
I_mark = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=(5, 6, 7)))
hat_R = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=(5, 6, 7)))
R = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=(5, 6, 7)))
loss = tf.reduce_sum(I_mark * tf.math.pow(hat_R - R, 2))
print("I.shape = ", I_mark.shape, "hat_R.shape = ",
      hat_R.shape, "R.shape = ", R.shape)
print("loss = ", loss)
