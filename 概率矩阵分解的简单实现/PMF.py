import tensorflow as tf
import numpy as np

class PMF:
    def __init__(self, userNum, itemNum, hNum, Activation = "sigmoid"):
        self.NormalInitializers = tf.keras.initializers.RandomNormal(mean= 0.0, stddev = 0.05)
        self.regularizersL2 = tf.keras.regularizers.l2(l = 0.01)
        self.optimizer = tf.keras.optimizers.Adam()
        self.Activation = "sigmoid"
        self.MSE = tf.keras.losses.MeanSquaredError()
        self.U = tf.Variable(self.NormalInitializers(shape=(userNum, hNum)))
        self.V = tf.Variable(self.NormalInitializers(shape=(hNum, itemNum)))

    def predict(self):
        hatRMatrix = tf.linalg.matmul(self.U, self.V)
        if (self.Activation == "sigmoid"):
            hatRMatrix = tf.keras.activations.sigmoid(hatRMatrix)
        return hatRMatrix

    def trainStep(self, RMatrix, IMatrix):
        with tf.GradientTape() as tape:
            hatRMatrix = self.predict()
            loss = self.MSE(RMatrix*IMatrix, hatRMatrix*IMatrix) + self.regularizersL2(self.U) + self.regularizersL2(self.V)
        trainParameter = [self.U, self.V]
        gradients = tape.gradient(loss, trainParameter)
        self.optimizer.apply_gradients(zip(gradients, trainParameter))
        return loss

    def fit(self, RMatrix, IMatrix, EpochNum = 2):
        print ("开始训练!")
        for i in range(EpochNum):
            lossValue = self.trainStep(RMatrix, IMatrix)
            print ("Epoch: " + str(i) + "; loss: " + str(lossValue))
        print ("训练结束!")

    

RMatrix =np.load('R_mat_All.npy')
IMatrix =np.load('I_mat_All.npy')
userNum, itemNum = len(RMatrix), len(RMatrix[0])
model = PMF(userNum, itemNum, 30)
model.fit(RMatrix, IMatrix, 1)
print (model.predict())
