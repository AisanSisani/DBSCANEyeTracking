from sklearn.cluster import DBSCAN
import glob
import numpy as np
import cv2
from random import randint


def rescale_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def hex_to_rgb(hex):
    return tuple(int(hex[i:i + 2], 16) for i in (1, 3, 5))


def main():
    txt_files = glob.glob("Apple_fixation_dataset/*.txt")
    X = list()
    for file in txt_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                line = lines[i]
                features = line.split()
                x = int(features[3])
                y = int(features[4])
                X.append([x, y])

    clustering = DBSCAN(eps=30, min_samples=30).fit(np.array(X))

    # labels for each point
    labels = clustering.labels_

    # number of clusters
    n_clusters = len(set(labels))
    print("Number of Clusters:", n_clusters)
    print(set(labels))

    # get the image
    img = cv2.imread('APPLE.png')
    colors = ['#%06X' % randint(0, 0xFFFFFF) for i in range(n_clusters-1)]

    print(colors)
    for i in range(len(X)):
        point = X[i]
        label = labels[i]
        if label == -1:
            color = hex_to_rgb('414141')
        else:
            color = hex_to_rgb(colors[label])
        cv2.circle(img, point, 2, color, 2)

    resized_img = rescale_img(img, 60)
    cv2.imshow('image', resized_img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
