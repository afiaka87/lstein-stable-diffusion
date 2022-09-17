# Builds on work from Lincoln D. Stein @ (https://github.com/lstein/stable-diffusion/blob/master/scripts/dream.py)

# cog support by Clay Mullis aka @afiaka87, various refactorings for single-call use, rather than using events.
# Copyright (c) 20222 afiaka87 aka Clay Mullis, (https://github.com/afiaka870)

# Derived from source code carrying the following copyrights
# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import random

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

import sys

import torch

sys.path.append("k-diffusion")
import tempfile

import transformers
from pytorch_lightning import logging

from ldm.dream.pngwriter import PngWriter
from ldm.generate import Generate

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
from typing import List, Optional

# cog imports
from cog import BaseModel, BasePredictor, Input, Path

SAMPLER_CHOICES = [
    "ddim",
    "k_dpm_2_a",
    "k_dpm_2",
    "k_euler_a",
    "k_euler",
    "k_heun",
    "k_lms",
    "plms",
]

# from PIL import PngImagePlugin, Image
# def image_progress(sample, step):
#     if step < steps - 1 and progress_images and step % progress_interval == 0:
#         pil_image = self.diffusion_model.sample_to_image(sample)
#         image_name = self.current_outdir / f"{prefix}.{seed}.{step:04d}.png"
#         info = PngImagePlugin.PngInfo()
#         info.add_text("Dream", prompt)
#         info.add_text("sd-metadata", json.dumps(metadata_dict))
#         pil_image.save(image_name, pnginfo=info)
#         print(f"Saved progress image {image_name} at timestep {step}")


def print_gpu_info(device_id: int = 0):
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(device_id)
    info = nvmlDeviceGetMemoryInfo(h)
    total = info.total / 1024**3
    free = info.free / 1024**3
    used = info.used / 1024**3
    print(f"GPU {device_id}: {used:.2f}/{total:2f}GB used ({free:.2f}GB free)")


def parse_variation_pairs(with_variations):
    variation_pairs = []
    for part in with_variations.split(","):
        seed_and_weight = part.split(":")
        assert (
            len(seed_and_weight) == 2
        ), f"Variation format is `seed:weight,seed:weight` but got {part}"
        try:
            seed, weight = int(seed_and_weight[0]), float(seed_and_weight[1])
        except ValueError:
            raise ValueError(f"Variation parsing failed for {part}")
        variation_pairs.append((seed, weight))
    return variation_pairs

class ImageSeedOutput(BaseModel):
    image_path: Path
    seed: Optional[int]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_outdir = Path(tempfile.mkdtemp())
        self.current_outdir.mkdir(parents=True, exist_ok=True)

        # preload the model
        self.gfpgan_dir = "/root/.cache/gfpgan"

        self.file_writer = PngWriter(self.current_outdir)
        self.diffusion_model = Generate(
            model="stable-diffusion-1.4",  # TODO update this when stable-diffusion V1.5 is released
            full_precision=False,
            sampler_name="k_lms",
            # embedding_path=None,
            # ddim_eta=0,
        )

        self.diffusion_model.load_model()  # do the slow model initialization

    def predict(
        self,
        prompt: str = Input(description="Prompt text", default=None),
        iterations: int = Input(description="Iterations/image-count", default=4),
        steps: int = Input(
            description="Refinement steps per iteration", default=50, le=500, ge=5
        ),
        seed: int = Input(
            description="Seed for random number generator, use -1 for a random seed.",
            default=-1,
        ),
        variation_amount: float = Input(
            description="Variation amount", default=0.0, le=1.0, ge=0.0
        ),
        with_variations: str = Input(
            description="Combine two or more variations using format `seed:weight,seed:weight`",
            default=None,
        ),
        width: int = Input(description="Width of generated image", default=512),
        height: int = Input(description="Height of generated image", default=512),
        sampler_name: str = Input(
            description="Sampler to use. Ignored when using an init image, which requires ddim sampling in v1.4",
            default="k_lms",
            choices=[
                "ddim",
                "k_dpm_2_a",
                "k_dpm_2",
                "k_euler_a",
                "k_euler",
                "k_heun",
                "k_lms",
                "plms",
            ],
        ),
        cfg_scale: float = Input(
            description="How strongly the prompt influences the image",
            default=7.5,
            le=10.0,
            ge=1.0,
        ),
        seamless: bool = Input(
            description="Whether the generated image should tile", default=False
        ),
        # init image settings
        init_img: Path = Input(
            description="(init_img) an initial image to use", default=None
        ),
        init_mask: Path = Input(
            description="(init_img) an initial mask to use", default=None
        ),
        fit: bool = Input(
            description="(init_img) Fit the generated image to the initial image",
            default=False,
        ),
        strength: float = Input(
            description="(init_img) Strength, 0.0 preserves image exactly, 1.0 replaces it completely",
            default=0.0,
            le=1.0,
            ge=0.0,
        ),
        gfpgan_strength: float = Input(
            description="Strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely",
            default=0.0,
            le=1.0,
            ge=0.0,
        ),
        ddim_eta: float = Input(
            description="Image randomness (eta=0.0 means the same seed always produces the same image)",
            default=0.0,
            le=1.0,
            ge=0.0,
        ),
        embiggen: str = Input(
            description="Scale factor relative to the size of the --init_img (-I), followed by ESRGAN upscaling strength (0-1.0), followed by minimum amount of overlap between tiles as a decimal ratio (0 - 1.0) or number of pixels",
            default=None,
        ),
        embiggen_tiles: str = Input(
            description="List of tiles by number in order to process and replace onto the image e.g. `0 2 4`",
            default=None,
        ),
        upscale_strength: float = Input(
            description="Weight to use when upscaling.", default=None
        ),
        upscale_level: int = Input(
            description="How much to upscale (2x or 4x)", default=None, choices=[2, 4]
        ),
        skip_normalize: bool = Input(
            description="Skip normalization of the various weighted CLIP embeds",
            default=False,
        ),
        log_tokenization: bool = Input(description="Log tokenization", default=False),
    ) -> List[ImageSeedOutput]:
        """Generate an image from a prompt"""
        if seed <= -1:
            seed = random.randint(0, 2**32 - 1)

        # setup upscale param by concatenating the strength and level
        upscale = None
        if upscale_level is not None and upscale_strength is None:
            upscale = f"{upscale_level} {upscale_strength}"
            print(f"Upscale: {upscale}")

        variation_pairs = None
        if int(seed) > -1 and with_variations is not None:
            variation_pairs = parse_variation_pairs(with_variations)
            print(f"Using variations {variation_pairs}")
        
        outputs = [
            ImageSeedOutput(image_path=Path(image_path), seed=seed)
            for image_path, seed in self.diffusion_model.prompt2png(
                prompt,
                outdir=self.current_outdir,
                iterations=iterations,
                steps=steps,
                seed=seed,
                cfg_scale=cfg_scale,
                ddim_eta=ddim_eta,
                skip_normalize=skip_normalize,
                # image_callback=None,
                # step_callback=image_progress,
                width=width,
                height=height,
                sampler_name=sampler_name,
                seamless=seamless,
                log_tokenization=log_tokenization,
                with_variations=variation_pairs,
                variation_amount=variation_amount,
                # these are specific to img2img and inpaint
                init_img=str(init_img) if init_img else None,
                init_mask=str(init_mask) if init_mask else None,
                fit=fit,
                strength=strength,
                # these are specific to embiggen (which also relies on img2img args)
                embiggen=embiggen,
                embiggen_tiles=embiggen_tiles,
                # these are specific to GFPGAN/ESRGAN
                gfpgan_strength=gfpgan_strength,
                # save_original=True, deprecated (doesnt even get used)
                upscale=upscale,
                # Set this True to handle KeyboardInterrupt internally
                catch_interrupts=False,
            )
        ]
        print(f"Generated {len(outputs)} images")
        print("Max VRAM usage:")
        print_gpu_info()
        return outputs
