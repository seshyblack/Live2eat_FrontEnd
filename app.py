import datetime
import glob
import os
import pickle
import statistics
import time
from datetime import timedelta
from typing import Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import time

from food_frame_export import *
from food_frame_extract import *
from food_prediction import *
from food_video_selection import *

from google.cloud import storage
from google.cloud import videointelligence as vi
from google.oauth2 import service_account
from sklearn.cluster import KMeans
from PIL import Image

# specify cloud credentials
#---------------------------------------------------------------
st.set_page_config(
    page_title="Live2Eat",
    page_icon="🐍",
    layout="centered",  # wide
    initial_sidebar_state="auto")  # collapsed

# Page Background
#---------------------------------------------------------------
# CSS = """
# h1 {
#     color: red;
# }
# .stApp {
#     background-image: url(https://images.unsplash.com/photo-1488900128323-21503983a07e);
#     background-size: cover;
# }
# """
# st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

# page header
#---------------------------------------------------------------
'''
# Live2Eat Food Tracking
Take the hard work out of tracking your food

'''
st.markdown('#')

# video selection
#---------------------------------------------------------------
option = st.selectbox('Please select a video',
                      ('Video 1', 'Video 2', 'Video 3', 'Video 4', 'Video 5'))

if option == 'Video 1':  # Bak Chor Mee
    video_URL = 'https://www.youtube.com/watch?v=V4GR-TcqYkk'
if option == 'Video 3':  # Kaya Toast
    video_URL = 'https://www.youtube.com/watch?v=7R-iTYFaS6A'
if option == 'Video 2':  # Hokkien Mee
    video_URL = 'https://www.youtube.com/watch?v=3zH2Hw4EE_U'
if option == 'Video 4':  # Chilli Crab
    video_URL = 'https://www.youtube.com/watch?v=g--tLRttm18'
elif option == 'Video 5':  #Chicken Rice
    video_URL = 'https://www.youtube.com/watch?v=S3UJD08RrFQ'

st.markdown('#')

st.video(video_URL, format="video/mp4", start_time=0)

if option:
    st.subheader('Video Analysis in Progress........')
else:
    st.subheader('Please select a video')

st.markdown('#')

# specify cloud credentials
#---------------------------------------------------------------
credentials = service_account.Credentials.from_service_account_info(
    st.secrets['gcp_service_account'])

# specify local folder locations
#---------------------------------------------------------------

dir = os.getcwd()
foldername = "Live2eat_FrontEnd"

raw_data_dir = f'{dir}/drive/MyDrive/{foldername}/raw_data'
export_path = f'{dir}/drive/MyDrive/{foldername}/data/predict_images'

try:
    # creating a folder named data
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    if not os.path.exists(export_path):
        os.makedirs(export_path)
        

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# select the video and download it
#---------------------------------------------------------------
video_path = os.path.join(raw_data_dir, 'video.mp4')
print('Reading video from: ', video_path)
cam = cv2.VideoCapture(video_path)

video_uri = video_uri(option, credentials)
download_video_opencv(video_uri, video_path, credentials)



# googleVideointelligence API video frames extract
#---------------------------------------------------------------

results = track_objects(video_uri, credentials)

with open("results.p", "wb") as f:
    pickle.dump(results, f)

with open("results.p", "rb") as f:
    results = pickle.load(f)

food_entity_id = '/m/02wbm'
food_times = print_object_frames(results, food_entity_id)
food_times = sorted(set(food_times))[::5]

# video frames export
#---------------------------------------------------------------

print('Current Dir: ', os.getcwd())
capture_images(food_times, cam, raw_data_dir)

sorted_dishes = sorted(glob.glob(raw_data_dir + "/*.jpg"),
                       key=lambda s: int(s.split('/')[-1].split('.')[0]))

print(f'length of {sorted_dishes = }')

print(f'length of sorted_dish after glob: {len(sorted_dishes)}')
dishes = create_dish_list(sorted_dishes)
resized_dishes = create_resized_dish_list(dishes)
resized_dishes_2d = create_reshaped_dish_list(resized_dishes)
file_labels = dish_clustering_dataframe(resized_dishes_2d, sorted_dishes)
median_dish(file_labels, raw_data_dir, export_path)

# model predict dishes
#---------------------------------------------------------------

prediction = np.array(predict()[0])
dish_images = predict()[1]
prediction_argmax = prediction.argmax(1)

print(f'the filepaths of predicted dish_images is {dish_images}')

# map predict results to image, dish name, calories
#---------------------------------------------------------------

dish_names = np.array([
    'BAK CHOR MEE', 'CHICKEN RICE', 'CHILLI CRAB', 'HOKKIEN MEE', 'KAYA TOAST'
])  # based on data.class_indices imagedatagen

dish_calories = [
    '511 calories', '607 calories', '1560 calories', '617 calories',
    '196 calories'
]

predicted_dishes = dish_names[prediction_argmax]
print(f'the list of predicted_dishes is {predicted_dishes}')

food_dictionary = dict(zip(dish_names, dish_calories))
print(f'the food dictionary of predicted_dishes is {food_dictionary}')

predicted_calories = list(map(food_dictionary.get, predicted_dishes))
print(f'the calories of predicted dishes is {predicted_calories}')

dishes_predicted_list = list(
    zip(predicted_dishes, predicted_calories, dish_images))
print(
    f'the fill list of predicted dishes to display is {dishes_predicted_list}')

# Progress Bar
#---------------------------------------------------------------
'Analysing your food videos...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    # Update the progress bar with each iteration.
    latest_iteration.text(f'Model Iteration {i+1}')
    bar.progress(i + 1)
    time.sleep(0.1)
'...and now we\'re done!'

st.markdown('#')

# display results
#---------------------------------------------------------------
st.markdown('#')

st.title("Dishes detected")

cols = st.columns(len(dishes_predicted_list))

for i, (predicted_dishes, predicted_calories,
        dish_images) in enumerate(dishes_predicted_list):

    image_opened = Image.open(dish_images)
    cols[i].image(image_opened)
    cols[i].text(predicted_dishes)
    cols[i].text(predicted_calories)
    select = cols[i].checkbox('Select', key=i)

st.markdown('#')
st.markdown('#')

# log food selected
#---------------------------------------------------------------

if st.button('Submit'):
    st.success("Your food has been logged!")
