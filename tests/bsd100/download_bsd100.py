import zipfile
from urllib import request

path_bsd100 = r"https://github.com/titu1994/Super-Resolution-using-Generative-Adversarial-Networks/releases/download/v0.1/bsd100.zip"

print("Downloading BSD100 images")
filehandler, _ = request.urlretrieve(path_bsd100)

zip_file = zipfile.ZipFile(filehandler)

print("Extracting images")
zip_file.extractall()