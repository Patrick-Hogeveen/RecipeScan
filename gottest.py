from transformers import AutoModel, AutoTokenizer

tokenzier = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True,
low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenzier.eos_token_id)
model = model.eval().cuda()

image_file = 'IMG_20240922_152049.jpg'

res = model.chat(tokenzier, image_file, ocr_type='ocr')
print(res)
