import os
import random
from concurrent.futures import ThreadPoolExecutor
import cv2
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F


def load_yolo_data(image_dir, label_dir):
    """Загружает данные в формате YOLO и возвращает абсолютные координаты bbox"""
    image_paths = []
    annotations = []

    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            bboxes = []
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, bw, bh = map(float, parts)
                x1 = int((x_center - bw / 2) * w)
                y1 = int((y_center - bh / 2) * h)
                x2 = int((x_center + bw / 2) * w)
                y2 = int((y_center + bh / 2) * h)
                bboxes.append([x1, y1, x2, y2])

            if bboxes:
                image_paths.append(img_path)
                annotations.append(bboxes)

    return image_paths, annotations


class FaceDataset(Dataset):
    def __init__(self, image_paths, annotations, min_size=800, max_size=1333):
        self.image_paths = image_paths
        self.annotations = annotations
        self.min_size = min_size
        self.max_size = max_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        print("Сработал индекс " + str(idx))
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Масштабирование с сохранением пропорций
        scale = min(self.min_size / min(h, w), self.max_size / max(h, w))
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))

        # Конвертация в тензор
        img_tensor = F.to_tensor(img)

        # Масштабирование bbox
        bboxes = []
        for x1, y1, x2, y2 in self.annotations[idx]:
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)
            bboxes.append([x1, y1, x2, y2])

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.ones(len(bboxes), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]),
            "iscrowd": torch.zeros(len(bboxes), dtype=torch.int64)
        }

        return img_tensor, target


def train_with_threads(model, train_data, epochs=5, batch_size=3, num_workers=7, device='cpu'):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        indices = list(range(len(train_data)))
        random.shuffle(indices)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            print("Всего элементов: " + str(len(indices)))
            # Обрабатываем батчами
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]

                batch = list(executor.map(train_data.__getitem__, batch_indices))

                # Подготовка данных
                images = [item[0].to(device) for item in batch]
                targets = [{k: v.to(device) for k, v in item[1].items()} for item in batch]

                # Обучение
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                print("обработано " + str(batch_start + batch_size) + " элементов")

        print(f"Epoch {epoch + 1}, Loss: {losses.item():.4f}")


def get_model(num_classes=2):
    """Инициализация модели с правильными параметрами"""
    backbone = torchvision.models.mobilenet_v2(weights=None).features
    backbone.out_channels = 1280  # Для MobileNetV2

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        min_size=800,
        max_size=1333
    )
    return model


def get_trained_model(img_paths, labels_path, epochs=5):
    image_paths, annotations = load_yolo_data(img_paths, labels_path)
    train_data = FaceDataset(image_paths, annotations)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2)
    print("Модель создана")
    train_with_threads(model, train_data, epochs=epochs, device=device)
    return model

# Сериализация
def save_model(model, path):
    torch.save({'state_dict': model.state_dict()}, path)
    print("Модель CNN сериализована")


# Десериализация
def load_model(path, device='cpu'):

    checkpoint = torch.load(path, map_location=device)

    model = get_model()

    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval()


def predict(model, image_path, device='cpu'):
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Масштабирование как при обучении
    scale = min(800 / min(h, w), 1333 / max(h, w))
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))

    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    if len(boxes) > 0:
        boxes = boxes / scale  # Масштабируем обратно

    return boxes


def detect_and_draw(model, img_path, path_to_save):
    detections = predict(model, img_path)
    image = cv2.imread(img_path)

    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imwrite(path_to_save, image)