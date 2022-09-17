import torch
import warnings
import os
import sys
import numpy as np

from PIL import Image

# from scripts.dream import create_argv_parser
from ldm.dream.args import Args

opt = Args()
opt.parse_args()
opt.gfpgan_dir = os.path.expanduser("~/.cache/gfpgan/")
opt.gfpgan_model_path = "experiments/pretrained_models/GFPGANv1.3.pth"
assert opt.gfpgan_dir is not None, "GFPGAN directory not specified"
model_path = os.path.join(opt.gfpgan_dir, opt.gfpgan_model_path)
gfpgan_model_exists = os.path.exists(model_path)
assert gfpgan_model_exists, "GFPGAN model not found"


def run_gfpgan(image, strength, seed, upsampler_scale=4):
    print(f">> GFPGAN - Restoring Faces for image seed:{seed}")
    gfpgan = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        assert gfpgan_model_exists, "GFPGAN model not found"

        sys.path.append(os.path.abspath(opt.gfpgan_dir))
        from gfpgan import GFPGANer

        bg_upsampler = _load_gfpgan_bg_upsampler(
            opt.gfpgan_bg_upsampler, upsampler_scale, opt.gfpgan_bg_tile
        )

        gfpgan = GFPGANer(
            model_path=model_path,
            upscale=upsampler_scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
        )

    if gfpgan is None:
        print(f">> WARNING: GFPGAN not initialized.")
        print(
            f">> Download https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth to {model_path}, \nor change GFPGAN directory with --gfpgan_dir."
        )
        return image

    if image.mode != "RGB":
        image = image.convert("RGB")

    cropped_faces, restored_faces, restored_img = gfpgan.enhance(
        np.array(image, dtype=np.uint8),
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    res = Image.fromarray(restored_img)

    if strength < 1.0:
        # Resize the image to the new image if the sizes have changed
        if restored_img.size != image.size:
            image = image.resize(res.size)
        res = Image.blend(image, res, strength)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gfpgan = None

    return res


def _load_gfpgan_bg_upsampler(bg_upsampler, upsampler_scale, bg_tile=400):
    if bg_upsampler == "realesrgan":
        if not torch.cuda.is_available():  # CPU or MPS on M1
            use_half_precision = False
        else:
            use_half_precision = True

        model_path = {
            2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        }

        if upsampler_scale not in model_path:
            return None

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if upsampler_scale == 4:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
        if upsampler_scale == 2:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )

        bg_upsampler = RealESRGANer(
            scale=upsampler_scale,
            model_path=model_path[upsampler_scale],
            model=model,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=use_half_precision,
        )
    else:
        bg_upsampler = None

    return bg_upsampler


def real_esrgan_upscale(image, strength, upsampler_scale, seed):
    print(f">> Real-ESRGAN Upscaling seed:{seed} : scale:{upsampler_scale}x")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        upsampler = _load_gfpgan_bg_upsampler(
            opt.gfpgan_bg_upsampler, upsampler_scale, opt.gfpgan_bg_tile
        )
    output, img_mode = upsampler.enhance(
        np.array(image, dtype=np.uint8),
        outscale=upsampler_scale,
        alpha_upsampler=opt.gfpgan_bg_upsampler,
    )
    res = Image.fromarray(output)
    if strength < 1.0:
        # Resize the image to the new image if the sizes have changed
        if output.size != image.size:
            image = image.resize(res.size)
        res = Image.blend(image, res, strength)
    return res
