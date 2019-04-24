import numpy as np
import matplotlib.pyplot as plt

from MyImageDataGenerator import MyImageDataGenerator

# 参考：https://qiita.com/takurooo/items/c06365dd43914c253240
def show_imgs(imgs, row, col):
    if len(imgs) != (row * col):
        raise ValueError("Invalid imgs len:{} col:{} row:{}".format(len(imgs), row, col))

    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, img in enumerate(imgs):
        plot_num = i+1
        ax = fig.add_subplot(row, col, plot_num, xticks=[], yticks=[])
        ax.imshow(img)
    plt.show()

datagen = MyImageDataGenerator(
    rescale=1/255.0,
    mix_up_alpha=2,
    random_crop=(375, 375))

max_img_num = 12
imgs = []
for d in datagen.flow_from_directory("images", batch_size=1, target_size=(375, 500), 
                                     classes=None):
    # target_size = (height, width)なのに注意
    imgs.append(np.squeeze(d[0], axis=0))
    if (len(imgs) % max_img_num) == 0:
        break
show_imgs(imgs, row=4, col=3)
