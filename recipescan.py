import subprocess
import sys
from transformers import AutoModel, AutoTokenizer
import pathlib
from utils.cleanimg import clean

#Add support for scanning whole folders


def extract(image_path):

    tokenzier = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True,
    low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenzier.eos_token_id)
    model = model.eval().cuda()

    

    res = model.chat(tokenzier, str(image_path), ocr_type='format')
    return res


if __name__ == "__main__":
    file_path = pathlib.Path(sys.argv[1])

    subprocess.run(['python3', 'utils/cleanimg.py', file_path])
    image_path = pathlib.Path(file_path.parent, "CLEAN_"+file_path.name)
    print(extract(image_path=image_path))



