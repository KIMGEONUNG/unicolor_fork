#!/usr/bin/env python

import os
from PIL import Image
import numpy as np

from colorizer import Colorizer
from utils_func import draw_strokes
from pycomar.images import show3plt

#
# with gr.Row():
#   image_canvas = gr.Image(source="upload",
#                           tool="color-sketch",
#                           label="Canvas",
#                           interactive=True)
#   viewer_mask = gr.Image(interactive=False,
#                          label="Mask",
#                          shape=(256, 256)).style(height=400)
#   viewer_hint = gr.Image(interactive=False,
#                          label="Hint",
#                          visible=False,
#                          shape=(16, 16)).style(height=400)
#   viewer_hint4view = gr.Image(interactive=False,
#                               label="Hint View",
#                               shape=(16, 16)).style(height=400)

def main():
    device = 'cuda:0'
    ckpt_file = '../framework/checkpoints/unicolor_mscoco/mscoco_step259999'
    # Load CLIP and ImageWarper for text-based and exemplar-based colorization
    colorizer = Colorizer(ckpt_file,
                          device, [256, 256],
                          load_clip=True,
                          load_warper=True)

    # Default directory is in /framework/
    I_gray = Image.open('../sample/images/1.jpg').convert('L')
    # I_uncond = colorizer.sample(I_gray, [], topk=100)

    # Hint points are indexed under 16*16 grid
    points = [{'index': [6 * 16, 2 * 16], 'color': [171, 209, 247]}]
    point_img = draw_strokes(I_gray, [256, 256],
                             points)  # Only for visualization
    I_stk = colorizer.sample(I_gray, points, topk=100)
    I_stk_cat = Image.fromarray(
        np.concatenate(
            [np.array(point_img), np.array(I_stk)], axis=1))

    # show3plt([I_gray, I_uncond, I_stk_cat])


def tmp():
    I_gray: Image.Image = Image.open('../sample/images/1.jpg').convert('L')
    points = [{'index': [6 * 16, 2 * 16], 'color': [171, 209, 247]}]
    point_img = draw_strokes(I_gray, [256, 256], points)
    print(points)
    point_img.show()
    print(point_img)


if __name__ == "__main__":
    main()
