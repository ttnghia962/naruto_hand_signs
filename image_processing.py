import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    # Đọc ảnh từ đường dẫn
    img = cv2.imread(image_path)
    # Chuyển đổi ảnh từ BGR sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def preprocess_image(img_rgb):
    # Resize ảnh về kích thước mong muốn
    img_resized = cv2.resize(img_rgb, (224, 224))
    # Thực hiện bất kỳ tiền xử lý nào khác (tùy thuộc vào yêu cầu của bạn)
    # ...

    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    img_normalized = img_resized / 255.0
    return img_normalized

def display_image(img):
    # Hiển thị ảnh
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Đường dẫn đến ảnh của bạn
    image_path = 'path/to/your/image.jpg'

    # Tải ảnh
    img = load_image(image_path)

    # Hiển thị ảnh gốc
    display_image(img)

    # Tiền xử lý ảnh
    preprocessed_img = preprocess_image(img)

    # Hiển thị ảnh sau khi tiền xử lý
    display_image(preprocessed_img)

