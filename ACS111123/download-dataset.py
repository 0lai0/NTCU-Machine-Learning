import kagglehub
import shutil
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
shutil.copytree(path, "./", dirs_exist_ok=True)