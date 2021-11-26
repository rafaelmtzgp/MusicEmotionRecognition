from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot

# load the dataset
dataset = loadtxt('music_data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:4]
y = dataset[:,4]
# define the keras model
print(X)
print(y)
y = to_categorical(y,4)

def train():
    model = Sequential()
    model.add(Dense(15, input_dim=4, activation='relu'))
    model.add(Dense(30, activation='sigmoid'))
    #model.add(Dense(15, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    history = model.fit(X, y, epochs=3000, batch_size=25)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))
    return accuracy, history, model


accuracy, history, model = train()
floatarucy = int(accuracy * 1000)
modelname = 'musicModel' + str(floatarucy) + '.h5'
print(modelname)
model.save(modelname)

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.legend()
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.legend()
pyplot.show()