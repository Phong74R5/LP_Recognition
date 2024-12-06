from ultralytics import YOLO
import cv2


# Load mô hình YOLO
model = YOLO(r'C:\Users\PhongLH\OneDrive\code\projectFinalAI\src\NhanDienBienSo\runs\detect\train2\weights\best.pt')

# Đường dẫn ảnh gốc
image_path = r'C:\Users\PhongLH\OneDrive\code\projectFinalAI\Bike_back\images\10.jpg'

# Dự đoán kết quả trên ảnh
results = model(image_path)

# Đọc ảnh gốc
bienso = cv2.imread(image_path)

# Duyệt qua từng kết quả
for i, result in enumerate(results):
    img = result.plot()  # Ảnh có bounding box
    cv2.imshow("Bien so a", img)  # Hiển thị ảnh với bounding box

    # Duyệt qua từng bounding box
    for j, box in enumerate(result.boxes.data):
        x1, y1, x2, y2, conf, cls = box  # Tọa độ, độ tin cậy, class
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Chuyển sang số nguyên

        # Cắt vùng ký tự từ ảnh gốc
        character = bienso[y1:y2, x1:x2]


        # Hiển thị từng ký tự đã cắt
        cv2.imshow(f'Bien so thu {j+1}', character)

    cv2.waitKey(0)

# Đóng tất cả các cửa sổ hiển thị
cv2.destroyAllWindows()
