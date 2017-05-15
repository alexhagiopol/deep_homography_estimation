import models
import pickle

train_file = 'train.p'
valid_file = 'valid.p'

with open(train_file, mode='rb') as f:
    train = pickle.load(f)
with open(valid_file, mode='rb') as f:
    valid = pickle.load(f)

model = models.homography_regression_model()
model.summary()
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
print("X_train shape = ", X_train.shape)
print("y_train shape = ", y_train.shape)

h = model.fit(x=X_train, y=y_train, verbose=1, batch_size=128, nb_epoch=50, validation_split=0.3)