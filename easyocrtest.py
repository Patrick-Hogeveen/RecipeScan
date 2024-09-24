import pandas as pd
import easyocr
import cv2

img = cv2.imread('clean.jpg')

reader = easyocr.Reader(['en'])
result = reader.readtext(img,paragraph='False')
df=pd.DataFrame(result)


extracted_text = ('\r\n').join(df[1])
#print(df[1])
#print(result)
print(extracted_text)