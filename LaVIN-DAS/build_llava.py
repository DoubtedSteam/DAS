import torch

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from transformers.generation.streamers import TextIteratorStreamer

from PIL import Image

import requests
from io import BytesIO

# from cog import BasePredictor, Input, Path, ConcatenateIterator
import time
import subprocess
from threading import Thread

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

# url for the weights mirror
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"
# files to download from the weights mirrors
weights = [
    {
        "dest": "liuhaotian/llava-v1.5-7b",
        # git commit hash from huggingface
        "src": "llava-v1.5-7b/006818fc465ebda4c003c0998674d9141d8d95f8",
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model-00001-of-00003.bin",
            "pytorch_model-00002-of-00003.bin",
            "pytorch_model-00003-of-00003.bin",
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]
    },
    {
        "dest": "openai/clip-vit-large-patch14-336",
        "src": "clip-vit-large-patch14-336/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin"
        ],
    }
]

def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")

def download_weights(baseurl: str, basedest: str, files: list[str]):
    basedest = Path(basedest)
    start = time.time()
    print("downloading to: ", basedest)
    basedest.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = basedest / f
        url = os.path.join(REPLICATE_WEIGHTS_URL, baseurl, f)
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix == ".json":
                download_json(url, dest)
            else:
                subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)


"""Load the model into memory to make running multiple predictions efficient"""
for weight in weights:
    download_weights(weight["src"], weight["dest"], weight["files"])

# tokenizer, model, image_processor, context_len = load_pretrained_model("liuhaotian/llava-v1.5-7b", model_name="llava-v1.5-7b", model_base=None, load_8bit=False, load_4bit=False)