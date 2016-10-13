import zipfile
from urllib import request

path_set5 = r"https://github.com/titu1994/Super-Resolution-using-Generative-Adversarial-Networks/releases/download/v0.1/Set5.zip"

print("Downloading Set5 images")
filehandler, _ = request.urlretrieve(path_set5)

zip_file = zipfile.ZipFile(filehandler)

print("Extracting images")
zip_file.extractall()