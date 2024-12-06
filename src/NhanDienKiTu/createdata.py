import os
import numpy as np
import cv2

def process_data(input_path, label_mapping, output_images, output_labels):
    images = []
    labels = []

    for fi in os.listdir(input_path):
        label = label_mapping.get(fi, -1)
        if label == -1:
            print(f"Skipping unmatched file: {fi}")
            continue

        img_fi_path = os.listdir(os.path.join(input_path, fi))
        for img_path in img_fi_path:
            img_full_path = os.path.join(input_path, fi, img_path)
            img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {img_full_path}")
                continue

            img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
            img = img.reshape((28, 28, 1))
            images.append(img)
            labels.append(label)

    # Lưu dữ liệu
    images = np.array(images)
    labels = np.array(labels)
    np.save(output_images, images)
    np.save(output_labels, labels)
    print(f"Saved {len(images)} images and labels to {output_images}, {output_labels}")


# Map nhãn cho digits
digits_mapping = {
    "0": 21, "1": 22, "2": 23, "3": 24, "4": 25, "5": 26,
    "6": 27, "7": 28, "8": 29, "9": 30
}

# Map nhãn cho alphas
alphas_mapping = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "K": 8,
    "L": 9, "M": 10, "N": 11, "P": 12, "R": 13, "S": 14, "T": 15, "U": 16,
    "V": 17, "X": 18, "Y": 19, "Z": 20 
}

# Process digits
process_data("c:/Users/PhongLH/OneDrive/code/projectFinalAI/datasets/datakitu/digits", digits_mapping, r"C:\Users\PhongLH\OneDrive\code\projectFinalAI\src\NhanDienKiTu\data\digits_images.npy", r"C:\Users\PhongLH\OneDrive\code\projectFinalAI\src\NhanDienKiTu\data\digits_labels.npy")

# Process alphas
process_data("c:/Users/PhongLH/OneDrive/code/projectFinalAI/datasets/datakitu/alphas", alphas_mapping, r"C:\Users\PhongLH\OneDrive\code\projectFinalAI\src\NhanDienKiTu\data\alphas_images.npy", r"C:\Users\PhongLH\OneDrive\code\projectFinalAI\src\NhanDienKiTu\data\alphas_labels.npy")
