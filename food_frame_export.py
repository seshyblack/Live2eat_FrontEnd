import time
import statistics
import numpy as np
import pandas as pd
import cv2
import numpy as np
import pickle
import glob
import os
import streamlit as st
import collections

from sklearn.cluster import DBSCAN
from typing import Optional, Sequence
from datetime import timedelta
from google.oauth2 import service_account
from sklearn.cluster import KMeans

from google.cloud import videointelligence as vi
from google.cloud import storage


def capture_images(food_times, cam, raw_data_dir):

    for f in os.listdir(raw_data_dir):
        os.remove(os.path.join(raw_data_dir, f))

    print('Deleted successfully:', raw_data_dir)

    print(f"{food_times = }")
    if cam.isOpened():

        for time in food_times:

            capture_time = time * 1000
            cam.set(cv2.CAP_PROP_POS_MSEC, capture_time)

            ret, frame = cam.read()

            if ret:

                name = './raw_data/' + str(capture_time) + '.jpg'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

            # Release all space and windows once done
        cam.release()


def create_dish_list(sorted_dishes):
    dishes = []
    print(f'length of sorted_dish: {len(sorted_dishes)}')

    for dish in sorted_dishes:
        dish_image = cv2.imread(dish)
        dishes.append(dish_image)

    return dishes


def create_resized_dish_list(dishes):
    resized_dishes = []

    for dish_image in dishes:
        scale_percent = 25
        width = int(dish_image.shape[1] * scale_percent / 100)
        height = int(dish_image.shape[0] * scale_percent / 100)
        dsize = (width, height)
        # resize image
        resized_image = cv2.resize(dish_image, dsize)
        resized_dishes.append(resized_image)

    return resized_dishes


def create_reshaped_dish_list(resized_dishes):
    img = resized_dishes[0]
    resized_dishes_2d = np.array(resized_dishes).reshape(
        len(resized_dishes), img.shape[0] * img.shape[1] * img.shape[2])
    return resized_dishes_2d


def dish_clustering_dataframe_dbscan(resized_dishes_2d, sorted_dishes):

    def check_purity(data, eps, noise_threshold=0.05):
        clusters = DBSCAN(eps=eps, min_samples=2).fit(data)
        return collections.Counter(clusters.labels_)[-1] / len(
            clusters.labels_) < noise_threshold

    # tweaked from https://algo.monster/problems/binary_search_boundary
    def find_boundary(data):
        eps_range = range(10000, 50000, 1000)
        left, right = 0, len(eps_range) - 1
        boundary_index = -1

        while left <= right:
            mid = (left + right) // 2
            if check_purity(data, eps=eps_range[mid]):
                boundary_index = mid
                right = mid - 1
            else:
                left = mid + 1

        if boundary_index == -1:
            boundary_index = len(
                eps_range
            ) - 1  # get largest eps so all values go in same cluster

        return eps_range[boundary_index]

    eps_threshold = find_boundary(resized_dishes_2d)
    print(f'{eps_threshold = }')

    clusters = DBSCAN(eps=eps_threshold, min_samples=2).fit(resized_dishes_2d)

    file_labels = pd.DataFrame({
        'files': sorted_dishes,
        'labels': clusters.labels_
    })

    file_labels['time'] = (file_labels.files.str.split('/').str[-1].str.rsplit(
        '.', n=1).str[0].astype(float))
    return file_labels


# # Hanqi improved dish_clustering_dataframe
# def dish_clustering_dataframe(resized_dishes_2d, sorted_dishes):
#     K = 8
#     kmeans = KMeans(n_clusters=K, random_state=0)
#     clusters = kmeans.fit(resized_dishes_2d.astype('uint8'))
#     file_labels = pd.DataFrame({
#         'files': sorted_dishes,
#         'labels': clusters.labels_
#     })
#     # file_labels['time'] = file_labels.files.str.split('/').str[-1].str.split(
#     #     '.').str[0]
#     file_labels['time'] = (file_labels.files.str.split('/').str[-1].str.rsplit(
#         '.', n=1).str[0].astype(float))
#     return file_labels

## Hanqi original dish_clustering_dataframe
# def dish_clustering_dataframe(resized_dishes_2d, sorted_dishes):
#     K = 4
#     kmeans = KMeans(n_clusters=K, random_state=0)
#     clusters = kmeans.fit(resized_dishes_2d.astype('uint8'))
#     file_labels = pd.DataFrame({
#         'files': sorted_dishes,
#         'labels': clusters.labels_
#     })
#     file_labels['time'] = file_labels.files.str.split('/').str[-1].str.split(
#         '.').str[0]

#     return file_labels


def median_dish(file_labels, raw_data_dir, export_path):

    for f in os.listdir(export_path):
        os.remove(os.path.join(export_path, f))

    print('Deleted successfully:', export_path)

    median_image_times = file_labels.groupby('labels')['time'].apply(
        statistics.median_low)
    print(f'{median_image_times=}')
    for time in median_image_times:

        image = cv2.imread(f'{raw_data_dir}/{time}.0.jpg')
        cv2.imwrite(os.path.join(export_path, f'{time}.0.jpg'), image)
        print(f'{time}.0.jpg' + ' written to' +
              os.path.join(export_path, f'{time}.0.jpg'))
