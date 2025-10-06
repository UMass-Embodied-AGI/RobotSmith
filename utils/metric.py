import torch
import open_clip
import numpy as np
from PIL import Image


def clip_score(image, text):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # preprocess the image
    image = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Encode text and image
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokenizer([text]))

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity (CLIP score)
    clip_score = (image_features @ text_features.T).item()

    return clip_score

def chamfer_distance(x, y):
    x = x[:, None, :]
    y = y[None, :, :]
    dist = torch.cdist(x, y, p=2)
    dist_x_to_y = torch.min(dist, dim=1)[0]
    dist_y_to_x = torch.min(dist, dim=0)[0]
    return torch.mean(dist_x_to_y) + torch.mean(dist_y_to_x)

if __name__ == "__main__":
    paths = ['test_robot.png', 'test_gummy.png', 'test_bear.png']
    for image_path in paths:
        image = Image.open(image_path).convert("RGB")
        text = "teddy bear"
        score = clip_score(image, text)
        print(f"CLIP Score for {image_path}: {score}")