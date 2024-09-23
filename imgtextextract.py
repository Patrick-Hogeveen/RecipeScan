import pytesseract
import cv2
from PIL import Image



# Open the image file

#image = Image.open('IMG_20240922_152049.jpg')

image = cv2.imread('IMG_20240922_152049.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0) 



cv2.imwrite("clean.jpg", blur)

# Perform OCR using PyTesseract

text = pytesseract.image_to_string(blur)



# Print the extracted text

print(text)


