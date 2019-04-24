import os
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.utils import np_utils

from keras.applications.inception_v3 import InceptionV3
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from sklearn.model_selection import train_test_split

#import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10

from MyImageDataGenerator import MyImageDataGenerator

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
num_classes = len(classes)
image_size = 32

#InceptionV3の最小イメージサイズ
image_size_IV3 = 139

#メインの関数を定義する
def main():
    # MNISTデータ読み込み
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    #5x5枚の画像を表示する
#    plt.figure(figsize=(10,10))
#    for i in range(25):
#        rand_num=np.random.randint(0,50000)
#        plt.subplot(5,5,i+1)
#        plt.imshow(X_train[rand_num])
#        #x軸の目盛りを消す
#        plt.tick_params(labelbottom=False)
#        #y軸の目盛りを消す
#        plt.tick_params(labelleft=False)
#        #正解ラベルを表示
#        plt.title(y_train[rand_num])

    plt.show()
    
    #トレーニングデータをトレーニング用と検証用に分割
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.175)
    
    X_train = X_train.reshape(X_train.shape[0], image_size, image_size, 3)
    X_valid = X_valid.reshape(X_valid.shape[0], image_size, image_size, 3)
    X_test = X_test.reshape(X_test.shape[0], image_size, image_size, 3)
    
    X_train = X_train.astype("float32") / 255.0
    X_valid = X_valid.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

#    X_train = resize_image(X_train)
#    X_valid = resize_image(X_valid)
#    X_test = resize_image(X_test)
    
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_valid = np_utils.to_categorical(y_valid, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    model = model_train(X_train, y_train, X_test, y_test, X_valid, y_valid)
    model_eval(model, X_test, y_test)

#def resize_image(X):
#    num=len(X)
#    zeros = np.zeros((num,image_size_IV3,image_size_IV3,3))
#    for i, img in enumerate(X):
#        zeros[i] = cv2.resize(
#                img,
#                dsize = (image_size_IV3,image_size_IV3)
#                )
#        result = zeros
#    
#    del zeros
#    return result

def model_train(X_train, y_train, X_test, y_test, X_valid, y_valid):
    #InceptionV3の読み込み（全結合層なし、ImageNetで学習した重み使用)
    base_model = InceptionV3(
            include_top = False,
            weights = "imagenet",
            input_shape = None
            )

    #InceptionV3の図の緑色の部分（FC層）の作成
    top_model = Sequential()
    top_model.add(GlobalAveragePooling2D())
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='softmax'))
    
    #InceptionV3とFC層を結合してモデルを作成
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    #Data Augmentation（データ拡張）
#    datagen = ImageDataGenerator(
    datagen = MyImageDataGenerator(
            featurewise_center = False, #データセット全体で各チャンネルごとの画素値の平均を0
            samplewise_center = False,  #サンプルごとの画素値の平均を0
            featurewise_std_normalization = False,  #データセット全体で各チャンネルごとの画素値の分散を1
            samplewise_std_normalization = False,   #サンプルごとの画素値の分散を1
            zca_whitening = False,      #白色化
            rotation_range = 0,         #-x°~x° の範囲でランダムに回転
            width_shift_range = 0.1,    #[-x*Width, x*Width]の範囲でランダムに左右平行移動
            height_shift_range = 0.1,   #[-x*Height, x*Height]の範囲でランダムに上下平行移動
            horizontal_flip = True,     #ランダムに左右反転
            vertical_flip = False       #ランダムに上下反転
            )
    
    #EarlyStopping
    #監視する値の変化が停止した時に訓練を終了
    early_stopping = EarlyStopping(
            monitor='val_loss',     #監視する値
            min_delta=0,            #この値よりも絶対値の変化が小さければ改善していないとみなす
            patience=10,            #指定したエポック数の間改善がないと訓練停止
            verbose=1
            )
    
    # ModelCheckpoint
    weights_dir='./weights/'
    if os.path.exists(weights_dir)==False:os.mkdir(weights_dir)
    model_checkpoint = ModelCheckpoint(
            weights_dir + "val_loss{val_loss:.3f}_IV3.hdf5",
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            period=3
            )
    
    #reduce learning rate
    #評価値の改善が止まった時に学習率を減らす
    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',     #監視する値
            factor=0.1,             #学習率を減らす割合
            patience=3,             #何エポック改善が見られなかったら学習率の削減を行うか
            verbose=1
            )
    
    # log for TensorBoard
    logging = TensorBoard(log_dir="log/")
    
    # 250層以降を学習させる
    for layer in model.layers[:249]:
        layer.trainable = False

    # Batch Normalization の freeze解除
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True
    
    for layer in model.layers[249:]:
        layer.trainable = True
    
    # layer.trainableの設定後に、必ずcompile
    model.compile(
            optimizer = Adam(),
            loss = 'categorical_crossentropy',
            metrics = ["accuracy"]
            )
    
    history = model.fit_generator(
            datagen.flow(X_train, y_train, batch_size = 32, resize = (image_size_IV3, image_size_IV3)),
            steps_per_epoch = X_train.shape[0] // 32,
            epochs = 20,
            validation_data = (X_valid, y_valid),
            callbacks = [early_stopping, reduce_lr, logging, model_checkpoint],
            shuffle = True,
            verbose = 1
            )
    
    #モデルを保存
    model.save('./cifar10_cnn_FineTuned_IV3.h5')
    
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
    
    