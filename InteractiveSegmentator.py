import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from matplotlib.backend_bases import MouseButton
from segment_anything import sam_model_registry, SamPredictor

class InteractiveSegmentator:

    def __init__(self, args):
        self.args = args
        self.predictor = self.prepare_sam("data/sam_vit_h_4b8939.pth")
        self.fnames = sorted([f for f in os.listdir(self.args.data_root) if f[-3:].lower() in ['png', 'jpg']])
        self.datalen = len(self.fnames)
        self.curr_idx = 0
        assert self.datalen > 0
        os.makedirs(os.path.join(self.args.data_root, "label"), exist_ok=True)
        os.makedirs(os.path.join(self.args.data_root, "label255"), exist_ok=True)

        self.f = self.fnames[self.curr_idx]
        image = self.get_image(self.f)

        self.predictor.set_image(image)

        H, W = image.shape[:2]
        self.current_mask = np.zeros((H, W, 4))
        self.zero_mask = np.zeros((H, W, 4))

        self.input_point = []
        self.input_label = []
        self.plots = []
        self.color = np.array([30/255, 144/255, 255/255, 0.6])
        
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        self.axim = self.ax.imshow(image)
        self.axim_mask = self.ax.imshow(self.zero_mask)
        self.text = self.fig.text(0, 0.95, f"1/{self.datalen}", size=20)

        self.cid0 = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid1 = self.fig.canvas.mpl_connect('key_press_event', self.onpress)

        plt.show()
    
    def prepare_sam(self, sam_checkpoint, model_type="vit_h", device="cuda"):
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        return SamPredictor(sam)
    
    def get_image(self, filename):
        image = cv2.imread(os.path.join(self.args.data_root, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def save_mask(self):
        save_path = os.path.join(self.args.data_root, "label", self.f)
        save_path255 = os.path.join(self.args.data_root, "label255", self.f)
        mask = self.mask[...,None].astype(np.uint8).repeat(3, -1)
        cv2.imwrite(save_path, mask)
        cv2.imwrite(save_path255, mask * 255)
    
    def onclick(self, event):
        ix, iy = int(event.xdata), int(event.ydata)

        self.input_point.append([ix, iy])
        if event.button is MouseButton.LEFT:
            self.input_label.append(1)
            marker_color = 'g'
        elif event.button is MouseButton.RIGHT:
            self.input_label.append(0)
            marker_color = 'r'
        
        mask, _, _ = self.predictor.predict(
            point_coords=np.array(self.input_point),
            point_labels=np.array(self.input_label),
            multimask_output=False,
        )
        self.mask = mask[0]
        mask_image = self.mask[...,None] * self.color.reshape(1, 1, -1)
        self.axim_mask.set_data(mask_image)

        self.plots.append(self.ax.scatter(ix, iy, marker='*', color=marker_color, s=375, edgecolor='white', linewidth=1.25))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def onpress(self, event):
        if event.key in ['enter', 'backspace']:
            
            if event.key == 'enter':
                self.save_mask()
                self.curr_idx += 1
                if self.curr_idx >= self.datalen:
                    self.fig.canvas.mpl_disconnect(self.cid0)
                    self.fig.canvas.mpl_disconnect(self.cid1)
                    plt.close(self.fig)
                    exit(0)
            elif event.key == 'backspace':
                self.curr_idx = max(self.curr_idx - 1, 0)
            
            self.f = self.fnames[self.curr_idx]
            image = self.get_image(self.f)
            self.predictor.set_image(image)

            self.input_point = []
            self.input_label = []

            self.axim.set_data(image)
            self.axim_mask.set_data(self.zero_mask)
            self.text.set_text(f"{self.curr_idx+1}/{self.datalen}")
            for p in self.plots: p.remove()
            self.plots = []
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,  default='data/bear/images')
    args = parser.parse_args()

    IS = InteractiveSegmentator(args)