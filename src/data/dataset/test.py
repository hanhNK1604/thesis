from PIL import Image
import numpy as np

# Đường dẫn file
image_path = "/home/ubuntu/thesis/data/isic/images/ISIC_0036337.jpg"
mask_path = "/home/ubuntu/thesis/data/isic/labels/ISIC_0036337.jpg"

# Đọc ảnh và mask
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")  # chuyển sang grayscale

# Chuyển sang numpy array
img_np = np.array(image)
mask_np = np.array(mask)

# Tạo mask nhị phân (0/1)
mask_binary = mask_np > 128  # vùng segment là True

# 👉 Che vùng được segment (làm đen)
img_np[mask_binary] = 255

# Chuyển lại sang ảnh
masked_img = Image.fromarray(img_np)
masked_img.save("masked_image.jpg")


print(mask_np) 
print(mask_np.shape)