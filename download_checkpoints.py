import gdown
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True) 

folder_url = "https://drive.google.com/drive/folders/1IUHfsqVyhDOxX92hcB6VdG5aQs7a-0db?usp=drive_link"


output_dir = "checkpoints"

gdown.download_folder(url=folder_url, output=output_dir, quiet=False, use_cookies=False)
