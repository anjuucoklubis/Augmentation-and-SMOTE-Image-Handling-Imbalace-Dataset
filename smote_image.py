# Code By : Anju Lubis
# SMOTE Image Dataset

import os
import numpy as np
from imblearn.over_sampling import SMOTE


# SMOTE 
# Note : Edit / Ubah Path train, validaiton dan test anda
#=========================================================================================
train_dir = 'dataset/split_apple/train/'
val_dir = 'dataset/split_apple/validation/'
test_dir = 'dataset/split_apple/test/'

img_height, img_width = 256, 256

# Note: class_map adalah apa saja class yang ada pada dataset yang ingin anda SMOTE
class_map = {
'Apple___Apple_scab': 0,
'Apple___Black_rot': 1,
'Apple___Cedar_apple_rust' : 2,
'Apple___healthy' : 3,}

train_images = []
train_labels = []

for subdir in os.listdir(train_dir):
    label = class_map[subdir]
    for file in os.listdir(os.path.join(train_dir, subdir)):
        img = load_img(os.path.join(train_dir, subdir, file), target_size=(img_height, img_width))
        img_array = img_to_array(img)
        train_images.append(img_array)
        train_labels.append(label)

train_images = np.array(train_images, dtype=np.float32)
train_labels = np.array(train_labels)

unique, counts = np.unique(train_labels, return_counts=True)
print("Class distribution of training set before oversampling:")
for c, count in zip(unique, counts):
    print(f"Class {c}: {count} samples")
train_images_flat = train_images.reshape(-1, img_height * img_width * 3)
 
smote = SMOTE()
train_images_resampled, train_labels_resampled = smote.fit_resample(train_images_flat, train_labels)
train_images_resampled = train_images_resampled.reshape(-1, img_height, img_width, 3)
unique, counts = np.unique(train_labels_resampled, return_counts=True)
print("\nClass distribution of training set after oversampling:")
for c, count in zip(unique, counts):
    print(f"Class {c}: {count} samples")

val_images = []
val_labels = []

for subdir in os.listdir(val_dir):
    label = class_map[subdir]

    for file in os.listdir(os.path.join(val_dir, subdir)):
        img = load_img(os.path.join(val_dir, subdir, file), target_size=(img_height, img_width))
        img_array = img_to_array(img)
        val_images.append(img_array)
        val_labels.append(label)

val_images = np.array(val_images, dtype=np.float32)
val_labels = np.array(val_labels)
unique, counts = np.unique(val_labels, return_counts=True)
print("Class distribution of validation set before oversampling:")
for c, count in zip(unique, counts):
    print(f"Class {c}: {count} samples")

val_images_flat = val_images.reshape(-1, img_height * img_width * 3)

smote = SMOTE()
val_images_resampled, val_labels_resampled = smote.fit_resample(val_images_flat, val_labels)
val_images_resampled = val_images_resampled.reshape(-1, img_height, img_width, 3)
unique, counts = np.unique(val_labels_resampled, return_counts=True)
print("\nClass distribution of Validation set after oversampling:")
for c, count in zip(unique, counts):
    print(f"Class {c}: {count} samples")
    

test_images = []
test_labels = []

for subdir in os.listdir(test_dir):
    label = class_map[subdir]

    for file in os.listdir(os.path.join(test_dir, subdir)):
        img = load_img(os.path.join(test_dir, subdir, file), target_size=(img_height, img_width))
        img_array = img_to_array(img)
        test_images.append(img_array)
        test_labels.append(label)

test_images = np.array(test_images, dtype=np.float32)
test_labels = np.array(test_labels)
unique, counts = np.unique(test_labels, return_counts=True)
print("Class distribution of test set before oversampling:")
for c, count in zip(unique, counts):
    print(f"Class {c}: {count} samples")

test_images_flat = test_images.reshape(-1, img_height * img_width * 3)

smote = SMOTE()
test_images_resampled, test_labels_resampled = smote.fit_resample(test_images_flat, test_labels)
test_images_resampled = test_images_resampled.reshape(-1, img_height, img_width, 3)
unique, counts = np.unique(test_labels_resampled, return_counts=True)
print("\nClass distribution of test set after oversampling:")
for c, count in zip(unique, counts):
    print(f"Class {c}: {count} samples")