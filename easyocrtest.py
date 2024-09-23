import pandas as pd
import easyocr
import cv2

img = cv2.imread('IMG_20240922_152049.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noise=cv2.medianBlur(gray,3)
thresh = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
reader = easyocr.Reader(['en'])
result = reader.readtext(img,paragraph='False')
df=pd.DataFrame(result)


extracted_text = ('\r\n').join(df[1])
print(df[1])
print(result)
print(extracted_text)