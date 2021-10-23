import scipy.io
import random
from PIL import Image
import numpy as np

import pandas as pd 

labels = scipy.io.loadmat("./clothing-co-parsing/label_list.mat")["label_list"]

classes = ["null", "clothes", "hair", "skin"]
labels_to_classes = [1] * labels.shape[1]
labels_to_classes[41] = 3
labels_to_classes[19] = 2
labels_to_classes[0] = 0

print(labels)
colors = []
class_id = []
for i in range(4):
    id = [0] * 4
    id[i] = 1
    class_id.append(id)



def cvt_mat(name):
    truth = scipy.io.loadmat("./clothing-co-parsing/annotations/pixel-level/{}.mat".format(name))["groundtruth"]
    pixels = []
    h,w = truth.shape
    for c in range(h):
        row = []
        for r in range(w):
            row.append(class_id[labels_to_classes[truth[c][r]]])
        pixels.append(row)

    # print("doneA")
    pixels = np.array(pixels, dtype=np.uint8)
    # print(pixels)
    # print("doneB")
    pixels = np.reshape(pixels, (-1,h*4))
    # print(w*4)
    # print("doneC")
    pd.DataFrame(pixels).to_csv("./clothing-co-parsing/labels/{}.csv".format(name), header=None, index=None)
    # print("doneD")
    # np.savetxt("./clothing-co-parsing/labels/{}.csv".format(name), pixels, delimiter=",")

    # new_image = Image.fromarray(pixels)
    # new_image.save("./clothing-co-parsing/labels/{}.png".format(name))




for i in range(1,1004):
    print("Converting image {img:04d}".format(img=i))
    cvt_mat(f'{i:04}')


#print(colors)