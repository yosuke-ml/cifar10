import os, re
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import numpy as np

import matplotlib.pyplot as plt

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
model = load_model('./cifar10_cnn_FineTuned_VGG16.h5')

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
    
test_images =[]     #テスト用画像配列
true_classes = []   #正解ラベル配列
pred_classes = []   #予測結果ラベル配列
pred_probs = []     #予測確率配列

for picture in list_pictures('./test_img/'):
    X = []
    img = img_to_array(
            load_img(picture, target_size=(32, 32), grayscale=False))
    X.append(img)

    X = np.asarray(X)
    X = X.astype('float32')
    X = X / 255.0
    
    #features = model.predict(X)
    
    #print('----------')
    #print(picture)
    #print(classes[features.argmax()])
    #print('----------')

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)

    #テスト用画像を配列に格納
    test_images.append(array_to_img(img))
    
    #ファイル名から正解ラベルを取得して配列に格納
    base, ext = os.path.splitext(os.path.basename(picture))
    base = ''.join(list(base)[:-1])
    true_classes.append(base)
    
    #予測結果ラベルを配列に格納
    pred_classes.append(classes[predicted])
    
    #予測確率を配列に格納
    pred_probs.append(percentage)
    
    print(picture)
    print("{0} ({1} %)".format(classes[predicted], percentage))

correct_count = 0

#テスト用画像と予測ラベルを表示
plt.figure(figsize = (10, 5))
for i in range(len(test_images)):
    plt.subplot(3, 10, i + 1)
    plt.axis("off")
    if pred_classes[i] == true_classes[i]:
        correct_count += 1
        plt.title(pred_classes[i] + '\n' + str(pred_probs[i]))
    else:
        plt.title(pred_classes[i] + '\n' + str(pred_probs[i]), color = "red")
    plt.imshow(test_images[i])
    
plt.show()
    
#正解率の表示
print("======================================")
print("Accracy Rate: {0:.0%}".format(correct_count/len(test_images)))
    
