#!/usr/bin/env python

from colorizer import Colorizer
import gradio as gr
import os
from PIL import Image
import numpy as np
from cv2 import cv2
from skimage.color import rgb2gray
from skimage.transform import rescale, resize


def predict(img, hints):
    img = Image.fromarray(img)

    device = 'cuda:0'
    # ckpt_file = '../framework/checkpoints/unicolor_mscoco/mscoco_step259999'
    ckpt_file = '../framework/checkpoints/unicolor_imagenet/imagenet_step142124.ckpt'
    # Load CLIP and ImageWarper for text-based and exemplar-based colorization
    colorizer = Colorizer(ckpt_file,
                          device, [256, 256],
                          load_clip=True,
                          load_warper=True)

    I_gray = img.convert('L')

    points = []
    for i in range(16):
        for j in range(16):
            color = hints[i][j]
            if not np.array_equal(color, [0, 0, 0]):
                points.append({'index': [i * 16, j * 16], 'color': color})

    # Hint points are indexed under 16*16 grid
    # points = [{'index': [6 * 16, 2 * 16], 'color': [171, 209, 247]}]
    rgb, rgb_resize, rgb_luma_enhance = colorizer.sample_gui(I_gray,
                                                             points,
                                                             topk=100)

    return rgb_resize, rgb_luma_enhance


def gen_mask(x: np.ndarray):
    if x is None:
        return None
    mask = (x[..., 0] == x[..., 1]) & (x[..., 1] == x[..., 2]) & (
        x[..., 2] == x[..., 0]) == False
    x = mask[..., None] * x
    x = cv2.resize(x, (16, 16), interpolation=cv2.INTER_NEAREST)
    return x


def overlay_hint(img: Image.Image,
                 mask: Image.Image,
                 alpha=0.5) -> Image.Image:
    if img is None or mask is None:
        return None
    img = Image.fromarray(img)
    mask = Image.fromarray(mask)

    mask = mask.resize(img.size, Image.NEAREST)

    img = np.array(img)
    if len(img.shape) == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.tile(img, 3)

    mask = np.array(mask)
    bg = np.all(mask == [0, 0, 0], axis=-1)[..., None]
    fg = bg == False

    bg = img * bg
    fg = ((alpha * mask + (1 - alpha) * img) * fg).astype('uint8')

    blend = bg + fg
    blend = Image.fromarray(blend)

    return blend


def gui():
    path_ex = [os.path.join("images", p) for p in sorted(os.listdir("images"))]
    height_img = 300
    css = r"img { image-rendering: pixelated; }"

    with gr.Blocks(css=css) as demo:
        with gr.Box():  # Input control pannel
            with gr.Column():
                with gr.Row():
                    view_color = gr.Image(
                        label="RGB", interactive=True).style(height=height_img)

                    view_paint = gr.Image(label="Paint",
                                          shape=[256, 256],
                                          tool="color-sketch",
                                          interactive=True)
                with gr.Row():
                    view_gray = gr.Image(
                        interactive=False,
                        label="Gray",
                    ).style(height=height_img)
                    view_paint_copy = gr.Image(
                        interactive=False,
                        label="Paint(View)",
                    ).style(height=height_img)
                    view_mask = gr.Image(
                        label="Mask",
                        interactive=False).style(height=height_img)
                    view_overlay = gr.Image(
                        label="Overlay",
                        interactive=False).style(height=height_img)
        with gr.Box():
            with gr.Column():
                with gr.Row():  # Output control pannel
                    view_output_resize = gr.Pil(
                        label="Output(RGB resized)",
                        interactive=False).style(height=height_img)
                    view_output_luma_enhance = gr.Pil(
                        label="Output(Luma-enhancement)",
                        interactive=False).style(height=height_img)
                    view_output_gray = gr.Pil(
                        label="Output(Gray of RGB)",
                        interactive=False).style(height=height_img)
                btn = gr.Button("Colorize")
        gr.Examples(examples=[
            "images/ex1_ccrop_0256.jpg", "images/ex2_ccrop_0256.jpg",
            "images/ex1_resize_256_ccrop_0256.jpg",
            "images/ex2_resize_256_ccrop_0256.jpg"
        ],
                    inputs=view_color)

        # Events
        view_color.change(lambda x: rgb2gray(x) if x is not None else None,
                          inputs=view_color,
                          outputs=view_paint)
        view_color.change(lambda x: rgb2gray(x) if x is not None else None,
                          inputs=view_color,
                          outputs=view_gray)

        view_output_resize.change(lambda x: rgb2gray(x) if x is not None else None,
                                  inputs=view_output_resize,
                                  outputs=view_output_gray)

        view_paint.change(gen_mask, inputs=view_paint, outputs=view_mask)
        view_paint.change(lambda x: x if x is not None else None,
                          inputs=view_paint,
                          outputs=view_paint_copy)

        view_mask.change(overlay_hint,
                         inputs=[view_gray, view_mask],
                         outputs=view_overlay)

        btn.click(predict,
                  inputs=[view_color, view_mask],
                  outputs=[view_output_resize, view_output_luma_enhance])

    demo.launch(share=True)


if __name__ == "__main__":
    gui()
    # main()
