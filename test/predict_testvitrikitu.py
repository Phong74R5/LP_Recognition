# from ultralytics import YOLO
# import cv2

# # Load mô hình YOLO
# model = YOLO(r'C:\Users\PhongLH\OneDrive\code\projectFinalAI\src\NhanDienViTriKiTu\runs\detect\train5\weights\best.pt')

# # Đường dẫn ảnh gốc
# image_path = r'C:\Users\PhongLH\OneDrive\code\projectFinalAI\datakhungbienso\images\bien3.jpg'

# # Dự đoán kết quả trên ảnh
# results = model(image_path)

# # Đọc ảnh gốc
# bienso = cv2.imread(image_path)

# # Lấy tên class từ mô hình

# # Danh sách lưu thông tin bounding box và class
# characters = []

# # Duyệt qua từng kết quả
# for i, result in enumerate(results):
#     img = result.plot()  # Ảnh có bounding box
#     cv2.imshow("Bien so", img)  # Hiển thị ảnh với bounding box

#     # Duyệt qua từng bounding box
#     for j, box in enumerate(result.boxes.data):
#         x1, y1, x2, y2, conf, cls = box  # Tọa độ, độ tin cậy, class
#         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Chuyển sang số nguyên

#         # Cắt vùng ký tự từ ảnh gốc
#         character = bienso[y1:y2, x1:x2]
        

#         # Lưu thông tin vào danh sách
#         characters.append((j, conf, character))

# # Sắp xếp danh sách theo class name
# characters.sort(key=lambda x: x[0])

# # Hiển thị các ký tự theo thứ tự class
# for cls_name, conf, character in characters:
#     # Hiển thị từng ký tự đã cắt cùng với tên class
#     cv2.imshow(f'Ki tu {cls_name}', character)
#     print(f"Class = {cls_name}, Confidence = {conf:.2f}")

# cv2.waitKey(0)
# cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2

# Load mô hình YOLO
model = YOLO(r'C:\Users\PhongLH\OneDrive\code\projectFinalAI\src\NhanDienViTriKiTu\runs\detect\train5\weights\best.pt')

# Đường dẫn ảnh gốc
image_path = r'C:\Users\PhongLH\OneDrive\code\projectFinalAI\datakhungbienso\images\bien15.jpg'

# Dự đoán kết quả trên ảnh
results = model(image_path)

# Đọc ảnh gốc
bienso = cv2.imread(image_path)

# Danh sách lưu thông tin bounding box và class
characters = []

# Duyệt qua từng kết quả
for i, result in enumerate(results):
    img = result.plot()  # Ảnh có bounding box
    cv2.imshow("Bien so", img)  # Hiển thị ảnh với bounding box
    
    height, width, _ = img.shape
    threshold = int(0.5 * height)  # Ngưỡng để phân biệt dòng trên và dòng dưới (tùy chỉnh theo dữ liệu thực tế)
    # Duyệt qua từng bounding box
    for j, box in enumerate(result.boxes.data):
        x1, y1, x2, y2, conf, cls = box  # Tọa độ, độ tin cậy, class
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Chuyển sang số nguyên

        # Cắt vùng ký tự từ ảnh gốc
        character = bienso[y1:y2, x1:x2]

        # Lưu thông tin vào danh sách (thêm tọa độ x1, y1 để sắp xếp)
        characters.append((j, float(conf), character, x1, y1))

# Phân loại theo nhóm dòng trên (y nhỏ) và dòng dưới (y lớn)

group_y_small = [char for char in characters if char[4] < threshold]
group_y_large = [char for char in characters if char[4] >= threshold]

# Sắp xếp từng nhóm theo tọa độ x tăng dần
group_y_small.sort(key=lambda x: x[3])  # Sắp xếp theo x (index 3)
group_y_large.sort(key=lambda x: x[3])  # Sắp xếp theo x (index 3)

# Kết hợp hai nhóm: dòng trên trước, dòng dưới sau
sorted_characters = group_y_small + group_y_large

# Hiển thị các ký tự theo thứ tự đã sắp xếp
for cls_name, conf, character, x1, y1 in sorted_characters:
    cv2.imshow(f'Ki tu {cls_name}', character)
    print(f"Class = {cls_name}, Confidence = {conf:.2f}, X = {x1}, Y = {y1}")
print(threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
