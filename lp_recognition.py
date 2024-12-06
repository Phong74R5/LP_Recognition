from tensorflow.keras.models import load_model  # type: ignore
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import os

# Bảng ánh xạ ký tự
ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9'}


temp = './temp'
os.makedirs(temp, exist_ok=True)

# Đường dẫn các mô hình
weightBS = './src/NhanDienBienSo/runs/detect/train2/weights/best.pt'
weightVTKT = './src/NhanDienViTriKiTu/runs/detect/train5/weights/best.pt'
weightKT = './src/NhanDienKiTu/cnn_character_recognition.h5'

# Load các mô hình
modelBienSo = YOLO(weightBS)
modelVTKT = YOLO(weightVTKT)
modelKT = load_model(weightKT)

def load_image(image_path):
    """Tải ảnh và trả về dưới dạng numpy array."""
    img = tf.io.read_file(image_path)  # Đọc file ảnh
    img = tf.io.decode_image(img, channels=3)  # Giải mã ảnh
    return img.numpy()  # Trả về numpy array

def save_image(image_array, save_path):
    """Lưu ảnh từ numpy array."""
    tf.io.write_file(save_path, tf.io.encode_jpeg(image_array))

def resize_image(image_array, size):
    """Resize ảnh về kích thước mới."""
    return tf.image.resize(image_array, size).numpy()

def crop_image(image_array, box):
    """Cắt ảnh theo tọa độ khung."""
    x1, y1, x2, y2 = map(int, box)
    return image_array[y1:y2, x1:x2]

def preprocess_image(image_input):
    """Tiền xử lý ảnh từ file path hoặc numpy array."""
    if isinstance(image_input, str):  # Nếu đầu vào là đường dẫn file
        img = tf.io.read_file(image_input)  # Đọc file ảnh
        img = tf.io.decode_image(img, channels=1)  # Chuyển sang grayscale
    elif isinstance(image_input, np.ndarray):  # Nếu đầu vào là numpy array
        img = tf.convert_to_tensor(image_input, dtype=tf.float32)
        if img.shape[-1] == 3:  # Nếu ảnh có 3 kênh (RGB), chuyển sang grayscale
            img = tf.image.rgb_to_grayscale(img)
    else:
        raise ValueError("Input must be a file path or numpy array.")

    img = tf.image.resize(img, [28, 28])  # Resize ảnh
    img = img / 255.0  # Chuẩn hóa giá trị pixel về [0, 1]
    img = tf.expand_dims(img, axis=0)  # Thêm batch dimension: (1, 28, 28, 1)
    return img



def NhanDienKhungBS(img_path):
    """Nhận diện khung biển số."""
    bienso = load_image(img_path)
    results = modelBienSo(img_path)
    bienso_paths = []
    for i, result in enumerate(results):
        for j, box in enumerate(result.boxes.data):
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = crop_image(bienso, (x1, y1, x2, y2))
            bienso_path = os.path.join(temp, f"Bien{j}.jpg")
            save_image(cropped, bienso_path)
            bienso_paths.append(bienso_path)
    return bienso_paths

def NhanDienViTriKiTu(bienso_path, index):
    """
    Nhận diện vị trí ký tự từ biển số.

    Args:
        bienso_path (str): Đường dẫn đến ảnh biển số.
        index (int): Chỉ số nhận dạng ảnh.
        threshold (int): Ngưỡng phân loại dòng ký tự theo tọa độ y.

    Returns:
        List[str]: Danh sách chứa đường dẫn ảnh ký tự đã lưu.
    """
    # Đọc ảnh biển số
    khung = load_image(bienso_path)
    
        # Lấy kích thước ảnh
    height, width, _ = khung.shape
    threshold = int(0.3 * height)

    # Dự đoán kết quả
    results = modelVTKT(bienso_path)
    anhkitu_paths = []

    # Danh sách chứa thông tin ký tự
    characters = []

    # Duyệt qua từng kết quả và thu thập thông tin
    for i, result in enumerate(results):
        for j, box in enumerate(result.boxes.data):
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            cropped = crop_image(khung, (x1, y1, x2, y2))  # Cắt vùng ký tự
            characters.append((cropped, x1, y1))  # Lưu ảnh ký tự và tọa độ x1, y1

    # Phân loại ký tự thành dòng trên và dòng dưới
    group_y_small = [char for char in characters if char[2] < threshold]
    group_y_large = [char for char in characters if char[2] >= threshold]

    # Sắp xếp từng dòng theo tọa độ x tăng dần
    group_y_small.sort(key=lambda x: x[1])  # Sắp xếp theo x1
    group_y_large.sort(key=lambda x: x[1])  # Sắp xếp theo x1

    # Kết hợp hai nhóm: dòng trên trước, dòng dưới sau
    sorted_characters = group_y_small + group_y_large

    # Lưu từng ký tự đã sắp xếp
    for order, (cropped, x1, y1) in enumerate(sorted_characters):
        # Đường dẫn lưu ảnh ký tự
        anhkitu_path = os.path.join(temp, f"Bien{index}_{order}.jpg")
        save_image(cropped, anhkitu_path)
        anhkitu_paths.append(anhkitu_path)
    return anhkitu_paths

def PhanLoaiKiTu(anhkitu_path):
    """Phân loại ký tự từ ảnh."""
    # Kiểm tra và tiền xử lý
    processed_image = preprocess_image(anhkitu_path)
    # Dự đoán
    prediction = modelKT.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return ALPHA_DICT[predicted_class]

def cleanTemp():
    folder_path = "./temp"
    # Duyệt qua các tệp trong thư mục
    for file_name in os.listdir(folder_path):
        # Kiểm tra nếu tên tệp bắt đầu bằng 'Bien'
        if file_name.startswith(f"Bien"):
            file_path = os.path.join(folder_path, file_name)
            # Kiểm tra nếu là tệp trước khi xóa
            if os.path.isfile(file_path):
                os.remove(file_path)

# Main processing

def MainProcess(source_img):
    """Thực hiện dự đoán biển số."""
    Bienso_paths = NhanDienKhungBS(source_img)

    for index, BSpath in enumerate(Bienso_paths):
        KiTu_paths = NhanDienViTriKiTu(BSpath, index)
        string = ""
        for anhkitu_path in KiTu_paths:
            string += PhanLoaiKiTu(anhkitu_path)
        print(f"Bien so: {string}\n")
        return string
 

