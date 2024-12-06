from tensorflow.keras.models import load_model # type: ignore
import cv2
import numpy as np

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9'}

# Tải lại mô hình từ file
model = load_model(r'C:\Users\PhongLH\OneDrive\code\projectFinalAI\src\NhanDienKiTu\cnn_character_recognition.h5')

# Hàm xử lý ảnh đầu vào
def preprocess_image(image_path):
    # Đọc ảnh thực tế
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh ở chế độ grayscale
    img = cv2.resize(img, (28, 28))  # Resize ảnh về 28x28
    img = img / 255.0  # Chuẩn hóa giá trị pixel
    img = np.expand_dims(img, axis=-1)  # Thêm kênh vào cuối (28, 28, 1)
    img = np.expand_dims(img, axis=0)  # Thêm batch size (1, 28, 28, 1)
    return img

# Đường dẫn đến ảnh thực tế
image_path = r'C:\Users\PhongLH\OneDrive\code\projectFinalAI\datasets\datakitu\alphas\H\Screenshot 2024-11-17 114610.png'

# Xử lý ảnh
processed_image = preprocess_image(image_path)

# Dự đoán
prediction = model.predict(processed_image)
predicted_class = np.argmax(prediction)  # Lấy lớp dự đoán có xác suất cao nhất

predicted_character = ALPHA_DICT[predicted_class]

print(f"Ký tự dự đoán: {predicted_character}")


