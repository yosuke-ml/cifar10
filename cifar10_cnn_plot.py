import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)
image_size = 32

#メインの関数を定義する
def main():
    # MNISTデータ読み込み
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    #5x5枚の画像を表示する
    plt.figure(figsize=(10,10))
    for i in range(25):
        rand_num=np.random.randint(0,50000)
        plt.subplot(5,5,i+1)
        plt.imshow(X_train[rand_num])
        #x軸の目盛りを消す
        plt.tick_params(labelbottom='off')
        #y軸の目盛りを消す
        plt.tick_params(labelleft='off')
        #正解ラベルを表示
        plt.title(y_train[rand_num])

    plt.show()
    
    #トレーニングデータをトレーニング用と検証用に分割
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.175)
    
    X_train = X_train.reshape(X_train.shape[0], image_size, image_size, 3)
    X_valid = X_valid.reshape(X_valid.shape[0], image_size, image_size, 3)
    X_test = X_test.reshape(X_test.shape[0], image_size, image_size, 3)
    
    X_train = X_train.astype("float32") / 255.0
    X_valid = X_valid.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_valid = np_utils.to_categorical(y_valid, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    model = model_train(X_train, y_train, X_test, y_test, X_valid, y_valid)
    model_eval(model, X_test, y_test)

def model_train(X, y, X_test, y_test, X_valid, y_valid):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(image_size, image_size, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    #opt = keras.optimizers.adam()
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, metrics=['accuracy'])
    
    #学習モデル図の作成
    plot_model(model, to_file='model.png')
    
    history = model.fit(X, y, 
                        batch_size=32, 
                        epochs=20,
                        validation_data=(X_valid, y_valid))
    
    #モデルを保存
    model.save('./cifar10_cnn.h5')
    
    #modelに学習させた時の変化の様子をplot
    plot_history(history)
    
    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

def plot_history(history):
    #精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()
    
    #損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()
    
if __name__ == "__main__":
    main()
    
    