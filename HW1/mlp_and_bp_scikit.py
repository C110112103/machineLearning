import time
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np


start = time.time()

# 輸出特徵
hidden_size = 4
learning_rate = 0.1
epochs = 1000
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# 輸出標籤
Y = np.array([0, 1, 1, 0])

# 創建MLP模型
mlp = MLPClassifier(hidden_layer_sizes=(hidden_size,), activation='logistic',
                    solver='sgd', learning_rate_init=learning_rate, max_iter=epochs, random_state=42)

# 擬合模型
mlp.fit(X, Y)

# 預測
y_pred = mlp.predict(X)

# 準確率
accuracy = accuracy_score(Y, y_pred)

end = time.time()
print("準確率: ", accuracy)


print("運算時長為: ", (end - start), "秒")
