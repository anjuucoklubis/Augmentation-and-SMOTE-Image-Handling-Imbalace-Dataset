# Code By : Anju Lubis
# Augmentation Image Dataset

import os
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image
import shutil
from pathlib import Path

classes = ["Apple___Apple_scab",
          "Apple___Black_rot",
          "Apple___Cedar_apple_rust",
          "Apple___healthy"]

# jumlah gambar yang diinginkan setelah augmentasi
num_images = 7000

# daftar teknik augmentasi
augmentation_techniques = {
    "affine": iaa.Affine(rotate=(-25, 25)),
    "gaussian_blur": iaa.GaussianBlur(sigma=(0, 1.0)),
    "brightness": iaa.Multiply((0.5, 1.5)),
    "zoom": iaa.Affine(scale=(1.0, 1.5)),
    "channel_shift": iaa.ChannelShuffle(p=0.5),
    "shear": iaa.ShearX((0, 50)),
    "gaussian_noise": iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))
}

# loop untuk setiap kelas yang ingin di-augmentasi
for cls in classes:
    # path folder training
    train_path = f"dataset/augmentation_general_v1/{cls}/"

    # path folder untuk simpan augmented image
    augmented_path = f"dataset/augmentation_general_v1/{cls}_augmented"

    # path folder untuk simpan hasil augmentasi yang dipindahkan
    new_train_path = f"dataset/augmentation_general_v1/{cls}/"

    # membuat folder augmented_path jika belum ada
    if not os.path.exists(augmented_path):
        os.makedirs(augmented_path)
        print(f"Folder {augmented_path} berhasil dibuat.")

    # hitung jumlah gambar pada folder data asli
    num_train_images = len(os.listdir(train_path))

    # hitung selisih antara jumlah gambar yang diinginkan dengan jumlah gambar pada folder data asli
    num_augmented_images_needed = num_images - num_train_images

    # loop sampai jumlah gambar sudah sesuai
    while len(os.listdir(augmented_path)) < num_augmented_images_needed:
        # randomly choose index gambar
        idx = np.random.randint(num_train_images)

        # load gambar
        image_path = os.path.join(train_path, os.listdir(train_path)[idx])
        image = np.array(Image.open(image_path))

        # augmentasi gambar secara acak
        technique_name = np.random.choice(list(augmentation_techniques.keys()))
        technique = augmentation_techniques[technique_name]
        seq = iaa.Sequential([technique])
        aug_image = seq(images=[image])

        # simpan gambar augmented
        technique_count = len([f for f in os.listdir(augmented_path) if technique_name in f])
        new_file_name = f"{cls}_{technique_name}_{technique_count+1}.jpg"
        new_image_path = os.path.join(augmented_path, new_file_name)
        Image.fromarray(aug_image[0]).save(new_image_path)

print(f"""
Proses augmentasi telah selesai.
======================================================================================
======================================================================================
""")

print(f"""\n
Proses Pemindahan file gambar agmentasi ke folder kelas asli.
======================================================================================
======================================================================================
""")

# pindahkan gambar hasil augmentasi ke folder data asli
for cls in classes:
    augmented_path = Path("dataset/augmentation_general_v1") / f"{cls}_augmented"
    new_train_path = Path("dataset/augmentation_general_v1") / cls
    num_train_images = len(os.listdir(new_train_path))

    for i, file_name in enumerate(os.listdir(augmented_path)):
        # dapatkan nama file dan teknik augmentasi dari nama file
        file_name_parts = file_name.split("_")
        technique_name = file_name_parts[1]
        new_file_name = file_name
      
        try:
            shutil.copyfile(augmented_path / file_name, new_train_path / new_file_name)
            print(f"File {file_name} dipindahkan ke Folder {new_train_path}.")
        except Exception as e:
            print(f"Gagal memindahkan file {file_name}. Error: {str(e)}")
        
        
    print(f"""
    Berhasil memindahkan semua isi file pada folder {augmented_path}!
    ======================================================================================
    ======================================================================================
    """)   
            

    # hapus folder augmented_path
    try:
        shutil.rmtree(augmented_path)
        print(f"""
        Folder {augmented_path} berhasil dihapus.
        ======================================================================================
        ======================================================================================
        """)    
    except Exception as e:
        print(f"Gagal menghapus folder {augmented_path}. Error: {str(e)}")

print("Augmentasi selesai!")
