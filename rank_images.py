import os
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
from simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
from CLIP import clip

parser = ArgumentParser()
parser.add_argument("directory")
parser.add_argument("--clip_model_name", type=str, default="ViT-B/16")
parser.add_argument("-t", "--top-n", type=int, default=50)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_name_to_model_path = {
    "ViT-B/32": "models/sac_public_2022_06_29_vit_b_32_linear.pth",
    "ViT-B/16": "models/sac_public_2022_06_29_vit_b_16_linear.pth",
    "ViT-L/14": "models/sac_public_2022_06_29_vit_l_14_linear.pth",
}

model_name_to_dim = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768}

clip_model = clip.load(args.clip_model_name, jit=False, device=device)[0]
clip_model.eval().requires_grad_(False)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

# 512 is embed dimension for ViT-B/16 CLIP
model = AestheticMeanPredictionLinearModel(model_name_to_dim[args.clip_model_name])
model.load_state_dict(
    torch.load(model_name_to_model_path[args.clip_model_name])
)
model = model.to(device)

def get_filepaths(parentpath, filepaths):
    paths = []
    for path in filepaths:
        try:
            new_parent = os.path.join(parentpath, path)
            paths += get_filepaths(new_parent, os.listdir(new_parent))
        except NotADirectoryError:
            paths.append(os.path.join(parentpath, path))
    return paths

filepaths = get_filepaths(args.directory, os.listdir(args.directory))
scores = []
for path in tqdm(filepaths):
    # This is obviously a flawed way to check for an image but this is just
    # a demo script anyway.
    if path[-4:] not in (".png", ".jpg"):
        continue
    img = Image.open(path).convert('RGB')
    img = TF.resize(img, 224, transforms.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, (224,224))
    img = TF.to_tensor(img).to(device)
    img = normalize(img)
    clip_image_embed = F.normalize(
        clip_model.encode_image(img[None, ...]).float(),
        dim=-1)
    score = model(clip_image_embed)
    if len(scores) < args.top_n:
        scores.append((score.item(),path))
        scores.sort()
    else:
        if scores[0][0] < score:
            scores.append((score.item(),path))
            scores.sort(key=lambda x: x[0])
            scores = scores[1:]
            
for score, path in scores:
    print(f"{score}: {path}")
