from transformers import AutoModel, AutoTokenizer
import cv2
import argparse

image = cv2.imread('Images/IMG_20240922_152049.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0) 

tokenzier = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True,
low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenzier.eos_token_id)
model = model.eval().cuda()

image_file = 'Images/clean.jpg'

res = model.chat(tokenzier, image_file, ocr_type='format')
res = model.chat(tokenzier, image_file, ocr_type='format')
print(res)
