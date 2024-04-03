from sklearn import svm
import numpy as np
import time

start = time.time()

# 假設X和Y的值
X = np.array([[1, 2], [2, 3], [3, 4]])
Y = np.array([-1, 1, -1])

# 初始化SVM模型
svm_model = svm.SVC(kernel='linear', C=1.0)

# 訓練模型
svm_model.fit(X, Y)

end = time.time()

print("權重:", svm_model.coef_)
print("截距:", svm_model.intercept_)

print("運算時長為: ", (end - start), "秒")
