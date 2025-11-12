from PIL import Image
import os

# Thư mục chứa ảnh
folder_path = "/home/ubuntu/thesis/data/cvc-clinic/labels"

for filename in os.listdir(folder_path):
    
    file_path = os.path.join(folder_path, filename)

    with Image.open(file_path) as img:

        img_resized = img.resize((256, 256))
        
        img_resized.save(file_path)

print("Done")
