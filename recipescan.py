import subprocess
import sys
from transformers import AutoModel, AutoTokenizer
import pathlib

#Add support for scanning whole folders
file_path = pathlib.Path(sys.argv[1])

subprocess.run(['python3', 'utils/cleanimg.py', file_path])

tokenzier = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True,
low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenzier.eos_token_id)
model = model.eval().cuda()

image_path = pathlib.Path(file_path.parent, "CLEAN_"+file_path.name)

res = model.chat(tokenzier, str(image_path), ocr_type='format')
print(res)

