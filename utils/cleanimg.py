import cv2
from PIL import Image
import sys
import pathlib



# Open the image file
#This is not a good way of doing this, the file existence should be checked first to avoid redoing computations

file_path = pathlib.Path(sys.argv[1])

image = cv2.imread(file_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0) 

new_path = pathlib.Path(file_path.parent, "CLEAN_"+file_path.name)

try:

    cv2.imwrite(new_path, blur)
except FileExistsError:
    print(f'The file {new_path} already exists.')