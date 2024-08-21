import os
import torch
import torchvision
import ultralytics
import yaml
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as fun
import torchvision.transforms.functional as F
import random
import numpy as np
import PIL
from PIL import Image, ImageDraw
import datetime
from scipy.stats import entropy
import shutil
from ultralytics.models import YOLO
import sys
from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QApplication,
    QProgressBar,
)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QEventLoop
from PyQt5.QtGui import QCloseEvent
import time

if torch.cuda.is_available():
    torch.cuda.set_device(0)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


class MessageDialog(QDialog):
    def __init__(self, len_images, len_labels, parent=None):
        super().__init__(parent)

        self.len_images = len_images
        self.len_labels = len_labels
        self.message_index = 0
        self.final_message = "Training finished. \nPress button to close Dialog."
        self.messages = [
            f"Number of Images: {len_images}\nNumber of Labels: {len_labels}",
            f"Training in Process.\nPlease Wait.",
        ]

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_message)
        self.timer.start(5000)

    def init_ui(self):
        layout = QVBoxLayout()

        # Message Label
        self.label = QLabel(self.messages[self.message_index], self)
        layout.addWidget(self.label)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        # Close Button (initially hidden)
        self.close_button = QPushButton("Close", self)
        self.close_button.setVisible(False)
        self.close_button.clicked.connect(self.accept)  # Closes the dialog
        layout.addWidget(self.close_button)

        self.setLayout(layout)
        self.setWindowTitle("Training Progress")

    def update_message(self):
        self.message_index = (self.message_index + 1) % len(self.messages)
        self.label.setText(self.messages[self.message_index])

    def show_final_message(self):
        self.label.setText(self.final_message)
        self.timer.stop()
        self.progress_bar.setValue(100)
        self.close_button.setVisible(True)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def closeEvent(self, event: QCloseEvent):
        # Ensure that the training thread is stopped before closing the dialog
        if self.timer.isActive():
            self.timer.stop()
        event.accept()


class TrainingThread(QThread):
    update_progress = pyqtSignal(int)
    update_signal = pyqtSignal()

    def __init__(self, model, data_path):
        super().__init__()
        self.model = model
        self.data_path = data_path

    def run(self):
        total_epochs = 20
        for epoch in range(total_epochs):
            # Simulate the training process
            self.model.train(
                data=self.data_path,
                epochs=1,
                momentum=0.9,
                optimizer="SGD",
                batch=4,
                workers=4,
                weight_decay=0.0001,
            )
            self.update_progress.emit((epoch + 1) * 100 // total_epochs)
        self.model.save(os.path.join(os.getcwd(), "best_save.pt"))
        self.update_signal.emit()  # Signal to update the dialog when training is done


class SelectionDialog(QDialog):
    def __init__(self, total_images, parent=None):
        super().__init__(parent)

        self.total_images = total_images
        self.message_index = 0
        self.final_message = "Image Selection finished.\nPress button to close Dialog."
        self.messages = [
            f"Total Images: {total_images}",
            f"Selecting Images based on uncertainty scores.\nPlease Wait.",
        ]

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_message)
        self.timer.start(5000)

    def init_ui(self):
        layout = QVBoxLayout()

        # Message Label
        self.label = QLabel(self.messages[self.message_index], self)
        layout.addWidget(self.label)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        # Close Button (initially hidden)
        self.close_button = QPushButton("Close", self)
        self.close_button.setVisible(False)
        self.close_button.clicked.connect(self.accept)  # Closes the dialog
        layout.addWidget(self.close_button)

        self.setLayout(layout)
        self.setWindowTitle("Image Selection Progress")

    def update_message(self):
        self.message_index = (self.message_index + 1) % len(self.messages)
        self.label.setText(self.messages[self.message_index])

    def show_final_message(self):
        self.label.setText(self.final_message)
        self.timer.stop()
        self.progress_bar.setValue(100)
        self.close_button.setVisible(True)

    def update_progress(self, value):
        self.progress_bar.setValue(value)


class SelectionThread(QThread):
    update_progress = pyqtSignal(int)
    update_signal = pyqtSignal()

    def __init__(
        self, unlabeled, train_labeled_set, validation_labeled_set, model_consistency
    ):
        super().__init__()
        self.unlabeled = unlabeled
        self.train_labeled_set = train_labeled_set
        self.validation_labeled_set = validation_labeled_set
        self.model_consistency = model_consistency
        self.sampled_images = []

    def run(self):
        # Seed setting for reproducibility
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)

        self.model_consistency.image_files = os.listdir(self.unlabeled)
        self.model_consistency.uncertainty_scores = {}

        total_images = len(self.model_consistency.image_files)

        for i, image_file in enumerate(self.model_consistency.image_files):
            if (
                image_file in self.train_labeled_set
                or image_file in self.validation_labeled_set
            ):
                uncertainty_score = 0
            else:
                image_path = os.path.join(self.unlabeled, image_file)
                image = Image.open(image_path)
                uncertainty_score = self.model_consistency.get_uncertainty(
                    image, self.unlabeled
                )
            self.model_consistency.uncertainty_scores[image_file] = uncertainty_score

            # Update progress
            progress = (i + 1) * 100 // total_images
            self.update_progress.emit(progress)

        sorted_images = sorted(
            self.model_consistency.uncertainty_scores.items(), key=lambda x: x[1]
        )
        self.sampled_images = [image_file for image_file, _ in sorted_images]

        # Emit signal to indicate selection is complete
        self.update_signal.emit()

    def get_sampled_images(self):
        return self.sampled_images


class ModelConsistency:
    def __init__(self, train_path, classes):
        self.train_path_images = os.path.join(train_path, "images")
        self.train_path_labels = os.path.join(train_path, "labels")
        self.classes = classes
        self.model = YOLO("yolov8n.pt", verbose=False)
        self.data_path = os.path.join(os.getcwd(), "dataset.yaml")

    def select_images(self, unlabeled, train_labeled_set, validation_labeled_set):
        dialog = SelectionDialog(total_images=len(os.listdir(unlabeled)))
        loop = QEventLoop()

        self.selection_thread = SelectionThread(
            unlabeled, train_labeled_set, validation_labeled_set, self
        )
        self.selection_thread.update_progress.connect(dialog.update_progress)
        self.selection_thread.update_signal.connect(dialog.show_final_message)
        self.selection_thread.finished.connect(loop.quit)
        self.selection_thread.finished.connect(dialog.close)
        self.selection_thread.start()

        dialog.show()
        loop.exec_()

        # Retrieve the sampled images after the thread finishes
        sampled_images = self.selection_thread.get_sampled_images()
        return sampled_images

    def evaluate(self):
        metrics = self.model.val()
        return metrics.box.map50

    def train_model(self):
        dialog = MessageDialog(
            len([file for file in os.listdir(self.train_path_images)]),
            len([file for file in os.listdir(self.train_path_labels)]),
        )
        loop = QEventLoop()

        self.thread = TrainingThread(self.model, self.data_path)
        self.thread.update_signal.connect(dialog.show_final_message)
        self.thread.update_progress.connect(dialog.update_progress)
        self.thread.finished.connect(loop.quit)
        self.thread.finished.connect(dialog.close)
        self.thread.start()

        dialog.show()
        loop.exec_()

    def get_uncertainty(self, image_path, unlabeled):
        consistency1 = 0
        consistency2 = 0
        original_image_results = self.model(unlabeled, verbose=False)

        if len(original_image_results[0].boxes) == 0:
            return 2
        original_image_confs = original_image_results[0].boxes.conf
        original_image_boxes = original_image_results[0].boxes.xyxy
        original_image_labels = original_image_results[0].boxes.cls

        augs = ["flip", "cutout", "smaller_resize", "rotation"]

        augmented_images, augmented_boxes = self.precompute_augmented_images(
            image_path, original_image_boxes
        )
        for (
            aug_image,
            aug_boxes,
            aug_name,
        ) in zip(augmented_images, augmented_boxes, augs):
            aug_image_results = self.model(aug_image, verbose=False)
            aug_image_confs = aug_image_results[0].boxes.conf
            aug_image_boxes = aug_image_results[0].boxes.xyxy

            iou_max = 0
            for orig_box in aug_boxes:
                max_iou = 0
                for aug_box in aug_image_boxes:
                    iou = self.calculate_iou(orig_box, aug_box)
                    if iou > max_iou:
                        max_iou = iou
                    iou_max += max_iou
            avg_iou_max = iou_max / len(aug_boxes)
            consistency1 += avg_iou_max

            max_len = max(len(original_image_confs), len(aug_image_confs))
            original_image_confs_padded = np.pad(
                original_image_confs.cpu().numpy(),
                (0, max_len - len(original_image_confs)),
                mode="constant",
            )
            aug_image_confs_padded = np.pad(
                aug_image_confs.cpu().numpy(),
                (0, max_len - len(aug_image_confs)),
                mode="constant",
            )

            p = (original_image_confs_padded + aug_image_confs_padded) / 2.0
            js_divergence = (
                entropy(original_image_confs_padded, p)
                + entropy(aug_image_confs_padded, p)
            ) / 2.0
            consistency2 = 1 - js_divergence

            consistency1 /= len(augs)

            return consistency1 + consistency2

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[2])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 and y2 < y1:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area

    def precompute_augmented_images(self, image_path, original_image_boxes):
        flip_image, flip_boxes = self.HorizontalFlip(image_path, original_image_boxes)
        cutout_image = self.cutout(image_path, original_image_boxes, 2)
        resize_image, resize_boxes = self.resize(image_path, original_image_boxes, 0.8)
        rot_image, rot_boxes = self.rotate(image_path, original_image_boxes, 5)

        return [flip_image, cutout_image, resize_image, rot_image], [
            flip_boxes,
            resize_boxes,
            rot_boxes,
        ]

    def HorizontalFlip(self, image, bbox):
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
        height, width = image.shape[-2:]
        image = image.flip(-1)
        b = bbox.clone()
        b[:, [0, 2]] = width - bbox[:, [2, 0]]

        image = F.to_pil_image(image)

        return image, b

    def cutout(
        self,
        image,
        boxes,
        cut_num=2,
        fill_val=0,
        bbox_remove_thres=0.4,
        bbox_min_thres=0.1,
    ):
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
        device = boxes.device
        image = image.to(device)
        original_h = image.size(1)
        original_w = image.size(2)
        original_channel = image.size(0)

        count = 0
        for _ in range(50):
            cutout_size_h = random.uniform(0.05 * original_h, 0.2 * original_h)
            cutout_size_w = random.uniform(0.05 * original_w, 0.2 * original_w)

            left = random.uniform(0, original_w - cutout_size_w)
            right = left + cutout_size_w
            top = random.uniform(0, original_h - cutout_size_h)
            bottom = top + cutout_size_h
            cutout = torch.FloatTensor(
                [int(left), int(top), int(right), int(bottom)]
            ).to(device)

            overlap_size = self.intersect(cutout.unsqueeze(0), boxes)
            area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            ratio = overlap_size / area_boxes
            if (
                ratio.max().item() > bbox_remove_thres
                or ratio.max().item() < bbox_min_thres
            ):
                continue

            cutout_arr = torch.full(
                (original_channel, int(bottom) - int(top), int(right) - int(left)),
                fill_val,
            ).to(device)
            image[:, int(top) : int(bottom), int(left) : int(right)] = cutout_arr
            count += 1
            if count >= cut_num:
                break
        image = F.to_pil_image(image)

        return image

    def resize(self, image, boxes, ratio):
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)
        h = image.size(1)
        w = image.size(2)
        ow = int(w * ratio)
        oh = int(h * ratio)

        image = F.to_pil_image(image)
        image = F.to_tensor(image.resize((ow, oh), Image.BILINEAR))
        image = F.to_pil_image(image)

        return image, boxes * ratio

    def rotate(self, image, boxes, angle, device="cpu"):
        if isinstance(image, Image.Image):
            image = F.to_tensor(image)

        if not isinstance(image, Image.Image):
            image = F.to_pil_image(image)

        new_image = image.copy()
        new_boxes = boxes.clone().to(device)

        w = image.width
        h = image.height
        cx = w / 2
        cy = h / 2
        new_image = new_image.rotate(angle, expand=True)
        angle = np.radians(angle)
        alpha = np.cos(angle)
        beta = np.sin(angle)

        AffineMatrix = torch.tensor(
            [
                [alpha, beta, (1 - alpha) * cx - beta * cy],
                [-beta, alpha, beta * cx + (1 - alpha) * cy],
            ],
            device=device,
        )

        box_width = (boxes[:, 2] - boxes[:, 0]).reshape(-1, 1)
        box_height = (boxes[:, 3] - boxes[:, 1]).reshape(-1, 1)

        x1 = boxes[:, 0].reshape(-1, 1)
        y1 = boxes[:, 1].reshape(-1, 1)

        x2 = x1 + box_width
        y2 = y1

        x3 = x1
        y3 = y1 + box_height

        x4 = boxes[:, 2].reshape(-1, 1)
        y4 = boxes[:, 3].reshape(-1, 1)

        corners = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), dim=1)
        corners = corners.reshape(-1, 2)
        corners = corners.to(device)

        corners = torch.cat(
            (corners, torch.ones(corners.shape[0], 1, device=device)), dim=1
        )

        AffineMatrix = AffineMatrix.to(device)

        cos = np.abs(AffineMatrix[0, 0].cpu())
        sin = np.abs(AffineMatrix[0, 1].cpu())
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        AffineMatrix[0, 2] += (nW / 2) - cx
        AffineMatrix[1, 2] += (nH / 2) - cy

        rotate_corners = torch.mm(AffineMatrix.float(), corners.t()).t()
        rotate_corners = rotate_corners.reshape(-1, 8)

        x_corners = rotate_corners[:, [0, 2, 4, 6]]
        y_corners = rotate_corners[:, [1, 3, 5, 7]]

        x_min, _ = torch.min(x_corners, dim=1)
        x_min = x_min.reshape(-1, 1)
        y_min, _ = torch.min(y_corners, dim=1)
        y_min = y_min.reshape(-1, 1)
        x_max, _ = torch.max(x_corners, dim=1)
        x_max = x_max.reshape(-1, 1)
        y_max, _ = torch.max(y_corners, dim=1)
        y_max = y_max.reshape(-1, 1)

        new_boxes = torch.cat((x_min, y_min, x_max, y_max), dim=1)

        scale_x = new_image.width / w
        scale_y = new_image.height / h

        new_image = new_image.resize((w, h))
        new_boxes /= torch.tensor([scale_x, scale_y, scale_x, scale_y])
        new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
        new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
        new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
        new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)

        image = F.to_tensor(new_image).to(device)
        image = F.to_pil_image(image)

        return image, new_boxes

    def intersect(self, boxes1, boxes2):
        device = boxes1.device
        n1 = boxes1.size(0)
        n2 = boxes2.size(0)

        boxes1 = boxes1.to(device)
        boxes2 = boxes2.to(device)

        max_xy = torch.min(
            boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
            boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2),
        )

        min_xy = torch.max(
            boxes1[:, 2:].unsqueeze(1).expand(n1, n2, 2),
            boxes2[:, 2:].unsqueeze(0).expand(n1, n2, 2),
        )
        inter = torch.clamp(max_xy - min_xy, min=0)
        return inter[:, :, 0] * inter[:, :, 1]
