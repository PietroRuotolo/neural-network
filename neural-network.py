import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#dataset
mnist = tf.keras.datasets.mnist

#Normalizing array (Basically squishing pixel's number (255) to a number between 0 and 1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

def init_params():
  W1 = np.random.randn(784, 128) * np.sqrt(2/784)
  b1 = np.zeros(128)
  W2 = np.random.randn(128, 10) * np.sqrt(2/128)
  b2 = np.zeros(10)
  return W1, b1, W2, b2

def forward(x, W1, b1, W2, b2):
  z1 = x @ W1 + b1
  H = np.maximum(0, z1)
  z2 = H @ W2 + b2
  expZ = np.exp(z2 - z2.max(axis=1, keepdims=True))
  A2 = expZ / expZ.sum(axis=1, keepdims=True)
  return z1, H, A2

def one_hot(y):
  n = y.shape[0] # 28
  oh = np.zeros((n, 10)) # Matrix of 28 rows and 10 columns of 0s
  oh[np.arange(n), y] = 1 # Arange returns an array of y's elements count, if we have an array with [3,4,5,5,1] than arange will return [0,1,2,3,4]. So we're builduing some matches with arange (index of n positions) and y matrix, and attributing to then 1 as match result
  return oh

def cross_entropy(A2, y):
  n = y.shape[0]
  correct_probs = A2[np.arange(n), y] + 1e-9
  L = -np.mean(np.log(correct_probs))
  return L

def backward(X, y, Z1, H, A2, W2): #Calculates the error on each value, when we retrain our neuralnetwor we'll update the values with the results (go in the opposite direction of the errors)
  n = y.shape[0]

  dZ2 = A2 - one_hot(y)
  dW2 = (1/n) * H.T @ dZ2 # how much he influenced × the guilt that arrived: dW2= (n/1)​Ht(Influence) ⋅ dZ2 (guilt)
  db2 = (1/n) * np.sum(dZ2, axis=0)

  dZ1 = (dZ2 @ W2.T) * (Z1 > 0) #Return by W2  
  dW1 = (1/n) * X.T @ dZ1
  db1 = (1/n) * np.sum(dZ1, axis=0)
  return dW1, db1, dW2, db2

def train(X, y, epochs=100, lr=0.1):
  W1, b1, W2, b2 = init_params()
  for i in range(epochs):
    #Here we have forward
    Z1, H, A2 = forward(X ,W1, b1, W2, b2)
    #Here we have 
    dW1, db1, dW2, db2 = backward(X, y, Z1, H, A2, W2)
    #Updating values... weight - learning_tax * gradient
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr  * db2

    if i % 10 == 0:
      loss = cross_entropy(A2, y)
      print(f"Era {i}: loss = {loss:.4f}")
  return W1, b1, W2, b2

def accuracy(X, y, W1, b1, W2, b2):
  _,_,A2 = forward(X ,W1, b1, W2, b2)
  predictions = np.argmax(A2, axis=1)
  return np.mean(predictions == y)

X_train = x_train.reshape(-1, 784)
X_test  = x_test.reshape(-1, 784)

W1, b1, W2, b2 = train(X_train, y_train, epochs=200, lr=0.1)

print("Train Accuracy:", accuracy(X_train, y_train, W1, b1, W2, b2))
print("Test Accuracy: ", accuracy(X_test,  y_test,  W1, b1, W2, b2))

_,_,A2 = forward(X_test, W1, b1, W2, b2)
predictions = np.argmax(A2, axis=1)

plt.figure(figsize=(10,4))
for i in range(10):
  plt.subplot(2, 5, i+1)
  plt.imshow(X_test[i].reshape(28,28), cmap='gray')
  p_color = 'green' if predictions[i] == y_test[i] else 'red'
  plt.title(f'Predictions: {predictions[i]}', color=p_color)
  plt.axis('off')
plt.tight_layout()
plt.show()