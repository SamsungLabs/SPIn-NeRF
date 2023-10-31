import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--src_root", type=str, required=True)
parser.add_argument("--dst_root", type=str, required=True)
parser.add_argument("--H", type=int, default=0)
parser.add_argument("--W", type=int, default=0)
parser.add_argument("--num_data", type=int, default=100)

args = parser.parse_args()

os.makedirs(args.dst_root, exist_ok=True)
fnames = sorted(os.listdir(args.src_root))
fnames = [f for f in fnames if f[-3:].lower() in ['png', 'jpg']]
idx = np.round(np.linspace(0, len(fnames)-1, args.num_data)).astype(np.int32)
fnames = [f for i, f in enumerate(fnames) if i in idx]

H, W = cv2.imread(os.path.join(args.src_root, fnames[0])).shape[:2]
H_dest = args.H if args.H != 0 else H
W_dest = args.W if args.W != 0 else W
fr = max(H_dest / H, W_dest / W)
H, W = round(H*fr), round(W*fr)

for f in tqdm(fnames):
    img = cv2.imread(os.path.join(args.src_root, f))

    if fr != 1:
        img = cv2.resize(img, dsize=(W, H), interpolation=cv2.INTER_AREA)

    if H != H_dest or W != W_dest:
        Hc, Wc = (H-H_dest)//2, (W-W_dest)//2
        img = img[Hc:Hc+H_dest, Wc:Wc+W_dest]

    cv2.imwrite(os.path.join(args.dst_root, f), img)