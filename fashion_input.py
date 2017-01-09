'''
This python file is responsible for the image processing
'''

import cv2
import numpy as np
import pandas as pd


shuffle = True
localization = False
imageNet_mean_pixel = [103.939, 116.799, 123.68]
global_std = 68.76

IMG_ROWS = 64
IMG_COLS = 64


def get_image(path, x1, y1, x2, y2):
    '''
    :param path: image path
    :param x1: the upper left and lower right coordinates to localize the apparels
    :param y1:
    :param x2:
    :param y2:
    :return: a numpy array with dimensions [img_row, img_col, img_depth]
    '''
    img = cv2.imread(path)
    if localization is True:
        if (x1 < x2) and (y1 < y2) and (x2 <= img.shape[0]) and (y2 <= img.shape[1]):
            img = img[x1:x2, y1:y2, :]
        img = cv2.resize(img, (IMG_ROWS, IMG_COLS))
        assert img.shape == (IMG_ROWS, IMG_COLS, 3)
    else:
        img = cv2.resize(img, (IMG_ROWS, IMG_COLS))

    img = img.reshape(1, IMG_ROWS, IMG_COLS, 3)

    return img


def load_data_numpy(df):
    '''
    :param df: a pandas dataframe with the image paths and localization coordinates
    :return: the numpy representation of the images and the corresponding labels
    '''

    num_images = len(df)
    image_path_array = df['image_path'].as_matrix()
    label_array = df['category'].as_matrix()
    x1 = df['x1'].as_matrix()
    y1 = df['y1'].as_matrix()
    x2 = df['x2'].as_matrix()
    y2 = df['y2'].as_matrix()

    image_array = np.array([]).reshape(-1, IMG_ROWS, IMG_COLS, 3)
    adjusted_std = 1.0/np.sqrt(IMG_COLS * IMG_ROWS * 3)

    for i in range(num_images):
        img = get_image(image_path_array[i], x1=x1[i], y1=y1[i], x2=x2[i], y2=y2[i])
        flip_indicator = np.random.randint(low=0, high=2)
        if flip_indicator == 0:
            img[0, ...] = cv2.flip(img[0, ...], 1)

        image_array = np.concatenate((image_array, img))

    image_array = (image_array - imageNet_mean_pixel) / global_std

    # Convert to BGR image for pre-train vgg16
    assert image_array.shape[1:] == (IMG_ROWS, IMG_COLS, 3)
    # image_array = image_array.transpose((0, 3, 1, 2))

    return image_array, label_array


def prepare_df(path, usecols, shuffle=shuffle):
    '''
    :param path: the path of a csv file
    :param usecols: which columns to read
    :return: a pandas dataframe
    '''
    df = pd.read_csv(path, usecols=usecols)
    if shuffle is True:
        order = np.random.permutation(len(df))
        df = df.iloc[order, :]
    return df
