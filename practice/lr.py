import numpy as np

class LogisticRegression:
    def __init__(self, rate = 0.001, iter=1000) -> None:
        self.rate = rate
        self.iter = iter
        self.w = None
        self.b = None
        self.loss = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)
    
    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        A = self._sigmoid(z)
        return A

    def fit(self, X, y):
        n_samples, n_feautres = X.shape

        # init parameters
        self.w = np.zeros(n_feautres)
        self.b = 0

        # gradient descent
        for _ in range(self.iter):
            A = self.forward(X)
            self.loss.append(self.compute_loss(y, A))
            dz = A - y  # derivative of sigmoid and bce X.T*(A-y)
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            # update parameters
            self.w -= self.rate * dw
            self.b -= self.rate * db
    
    def predict(self, X):
        thr = .5
        y_hat = np.dot(X, self.w) + self.b
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > thr else 0 for i in y_predicted]

        return np.array(y_predicted_cls)
    


from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

regressor = LogisticRegression(rate=0.0001, iter=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
cm = confusion_matrix(np.asarray(y_test), np.asarray(predictions))
print("Confusion Matrix:\n", np.array(cm))