import os
import random
import shutil


def reduct_images_dataset(images_path, annotations_path, new_size):

    reduced_images_folder = 'reduced_images'
    reduced_labels_folder = 'reduced_annotations'

    if os.path.exists(reduced_images_folder) and os.path.exists(reduced_labels_folder):
        return reduced_images_folder, reduced_labels_folder

    os.makedirs(reduced_images_folder, exist_ok=True)
    os.makedirs(reduced_labels_folder, exist_ok=True)


    image_files = [f for f in os.listdir(images_path)]


    selected_images = random.sample(image_files, min(new_size, len(image_files)))
    selected_labels = [os.path.splitext(f)[0] + '.txt' for f in selected_images]


    for file in selected_images:
        shutil.copy(os.path.join(images_path, file), 'reduced_images')

    for file in selected_labels:
        shutil.copy(os.path.join(annotations_path, file), 'reduced_annotations')

    print("Сокарщенный датасет создан")
    print("Количество тренировочных единиц в данных - " + str(len(os.listdir('reduced_images'))))
    return 'reduced_images', 'reduced_annotations'
