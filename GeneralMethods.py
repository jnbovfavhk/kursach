import os

import cv2


# Отрисовка прямоугольников на изображении
def draw_detections(image, detections):
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Рисуем прямоугольник
    return image


# Возвращает список полных путей изображений и к каждому из них список координат лиц
def load_yolo_annotations(images_dir, annotations_dir):
    annotations = []
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    i = 0
    for image_file in image_files:
        i += 1
        print(i)

        # Получение размера изображения
        image = cv2.imread(image_file)

        height, width = image.shape[:2]

        # Путь к соответствующему файлу аннотаций
        annotation_file = os.path.basename(image_file).rstrip('.jpg').rstrip('.png') + '.txt'
        annotation_path = os.path.join(annotations_dir, annotation_file)

        if os.path.exists(annotation_path):
            image_annotations = []
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    # Разделение строки на элементы
                    parts = line.strip().split()
                    class_id = int(parts[0])  # ID класса
                    x_center = float(parts[1]) * width  # Координата x центра
                    y_center = float(parts[2]) * height  # Координата y центра
                    w = float(parts[3]) * width  # Ширина
                    h = float(parts[4]) * height  # Высота

                    # Рассчитываем координаты прямоугольника
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    image_annotations.append([x1, y1, x2, y2])  # Добавляем координаты

            annotations.append(image_annotations)  # Добавляем аннотации для изображения

    return image_files, annotations

# Метод скользящего окна
def sliding_window(image, step_size=None, min_window_size=None, max_window_size=None, aspect_ratio=(1, 2)):
    if max_window_size is None:
        max_window_size = (
        image.shape[1], image.shape[0])  # Установить максимальный размер окна равным размеру изображения

    if min_window_size is None:
        min_window_size = (image.shape[1] // 20, image.shape[0] // 20)

    if step_size is None:
        step_size = min_window_size[0]

    for window_height in range(min_window_size[1], max_window_size[1] + 1,
                               max_window_size[1] // 20):  # Увеличиваем высоту окна
        for window_width in range(min_window_size[0], max_window_size[0] + 1,
                                  max_window_size[0] // 20):  # Увеличиваем ширину окна

            # Находим соотношение сторон
            width_to_height_ratio = window_width / window_height
            height_to_width_ratio = window_height / window_width

            step_size = window_width
            # Проверяем, что хотя бы одно из соотношений в заданном интервале(где может находится лицо)
            if (aspect_ratio[0] <= width_to_height_ratio <= aspect_ratio[1]) or \
                    (aspect_ratio[0] <= height_to_width_ratio <= aspect_ratio[1]):
                for y in range(0, image.shape[0] - window_height + 1, step_size):
                    for x in range(0, image.shape[1] - window_width + 1, step_size):
                        print(f'x = {x}, y = {y}, width = {window_width}, height = {window_height}')
                        yield (x, y, image[y:y + window_height, x:x + window_width])