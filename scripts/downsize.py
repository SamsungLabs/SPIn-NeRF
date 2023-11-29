import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True)
parser.add_argument("--factor", type=int, default=2)
args = parser.parse_args()

src_root = os.path.join(args.data_root, "images")
dst_root = os.path.join(args.data_root, f"images_{args.factor}")
os.makedirs(dst_root, exist_ok=True)
os.makedirs(os.path.join(dst_root, "label"), exist_ok=True)
os.makedirs(os.path.join(dst_root, "label255"), exist_ok=True)
fnames = sorted(os.listdir(src_root))
fnames = [f for f in fnames if f[-3:].lower() in ['png', 'jpg']]
fr = 1 / args.factor

for f in tqdm(fnames):
    msk_f = f.replace("jpg", "png")
    img = cv2.imread(os.path.join(src_root, f))
    msk = cv2.imread(os.path.join(src_root, "label", msk_f))

    img = cv2.resize(img, dsize=(0,0), fx=fr, fy=fr, interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, dsize=(0,0), fx=fr, fy=fr, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(dst_root, f), img)
    cv2.imwrite(os.path.join(dst_root, "label", msk_f), msk)
    cv2.imwrite(os.path.join(dst_root, "label255", msk_f), msk * 255)