# Builds on work from Lincoln D. Stein @ (https://github.com/lstein/stable-diffusion/blob/master/scripts/dream.py)

# cog support by Clay Mullis aka @afiaka87, various refactorings for single-call use, rather than using events.
# Copyright (c) 20222 afiaka87 aka Clay Mullis, (https://github.com/afiaka870)

# Derived from source code carrying the following copyrights
# Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import sys

# this uses the k-diffusion in our local directory, to use a pip install, remove this line, and check cog.yaml for further instructions.
sys.path.append("k-diffusion")

import random
import tempfile

import torch
import transformers
from pytorch_lightning import logging

from ldm.dream.args import Args
from ldm.dream.pngwriter import PngWriter
from ldm.generate import Generate

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
from typing import List, Optional

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

from urllib.request import urlretrieve


def download_image(url: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        urlretrieve(url, f.name)
        return f.name


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
        # init image settings
        init_img: Path = Input(
            description="(init_img) an initial image to use", default=None
        ),
        init_mask: Path = Input(
            description="(init_img) an initial mask to use", default=None
        ),
        init_strength: float = Input(
            description="(init_img/outpaint) Strength, 0.0 preserves image exactly, 1.0 replaces it completely. defaults to 0.75 (0.83 when outpainting).",
            default=None,
            le=1.0,
            ge=0.0,
        ),
        init_fit: bool = Input(
            description="(init_img) Resize the image to the model dimensions. Default to False.",
            default=False,
        ),
        init_color: Path = Input(
            description="(init_img) Path to reference image for color correction (used for repeated img2img and inpainting)",
            default=None,
        ),
        outpaint_direction: str = Input(
            description="(init_img) Outpaint an init_image in a given direction - `top`, `bottom`, `left`, `right`",
            default=None,
            choices=["top", "bottom", "left", "right"],
        ),
        outpaint_pixels: int = Input(
            description="(init_img) Number of pixels to outpaint in the given direction",
            default=None,
            ge=0,
        ),
        variation_amount: float = Input(
            description="Variation amount. Specifies how different variations will look.",
            default=0.0,
            le=1.0,
            ge=0.0,
        ),
        with_variations: str = Input(
            description="(optional) Combine two or more variations using format `seed:weight,seed:weight,seed:weight`",
            default=None,
        ),
        num_images: int = Input(description="Iterations/image-count", default=1),
        steps: int = Input(
            description="Denoising steps per each generation", default=50, le=500, ge=5
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
            le=20.0,
            ge=0.0,
        ),
        seamless: bool = Input(
            description="Whether the generated image should tile", default=False
        ),
        facetool: str = Input(
            description="Select the face restoration AI to use: gfpgan, codeformer",
            default="gfpgan",
            choices=["gfpgan", "codeformer"],
        ),
        gfpgan_strength: float = Input(
            description="Strength for GFPGAN. 0.0 preserves image exactly, 1.0 replaces it completely",
            default=0.0,
            le=1.0,
            ge=0.0,
        ),
        codeformer_fidelity: float = Input(
            description="Takes values between 0 and 1. 0 produces high quality but low accuracy. 1 produces high accuracy but low quality.",
            default=0.86,
            le=1.0,
            ge=0.0,
        ),
        embiggen: str = Input(
            description="(e.g. '4 0.75 .02') Scale factor, upscaling strength (0-1.0), followed by minimum amount of overlap between tiles as a decimal ratio (0 - 1.0)",
            default=None,
        ),
        embiggen_tiles: str = Input(
            description="List of tiles by number in order to process and replace onto the image e.g. `0 2 4`",
            default=None,
        ),
        upscale_strength: float = Input(
            description="Weight for ESRGAN upscaling.", default=None, le=1.0, ge=0.0
        ),
        upscale_level: int = Input(
            description="How much to upscale (2x or 4x) with ESRGAN",
            default=None,
            choices=[2, 4],
        ),
        seed: int = Input(
            description="Seed for random number generator, use -1 for a random seed.",
            default=-1,
        ),
        embedding_path: Path = Input(  # TODO
            description="(experimental/wip) Path to a textual-inversion embedding (.bin) from huggingface SD-concepts https://huggingface.co/sd-concepts-library",
            default=None,
        ),
        command_mode: bool = Input(
            description="(advanced) Parse `prompt` for `-`/`--` options, ignoring all other inputs and validation. Also activated when prompt starts with `!dream` (without backticks).",
            default=False,
        ),
    ) -> List[ImageSeedOutput]:
        """Generate an image from a prompt"""


        if init_strength is None:
            if outpaint_direction is not None:
                init_strength = 0.83 # default for outpainting
            else:
                init_strength = 0.75 # default for all other cases

        if command_mode or prompt.startswith("!dream"):
            dream_args = Args()
            prediction_options = vars(dream_args.parse_cmd(prompt))
            prediction_options["prompt"] = (
                prediction_options["prompt"].replace("!dream", "").strip()
            )
            if prediction_options["init_img"].startswith("http"):
                prediction_options["init_img"] = download_image(
                    prediction_options["init_img"]
                )
            if prediction_options["init_url"].startswith("http"):
                prediction_options["init_img"] = download_image(
                    prediction_options["init_url"]
                )
            if prediction_options["init_mask"].startswith("http"):
                prediction_options["init_mask"] = download_image(
                    prediction_options["init_mask"]
                )
        else:
            # perform validation, etc
            if seed <= -1:
                seed = random.randint(0, 2**32 - 1)
                print(f"Using random seed: {seed}")

            if embiggen is not None or embiggen_tiles is not None:
                assert (
                    init_img is not None
                ), "Must provide an initial image when using embiggen"

            upscale = None
            # setup upscale param by concatenating the strength and level
            if upscale_level is not None and upscale_strength is None:
                upscale = f"{upscale_level} {upscale_strength}"
                print(f"Upscale: {upscale}")

            variation_pairs = None
            if int(seed) > -1 and with_variations is not None:
                variation_pairs = parse_variation_pairs(with_variations)
                print(f"Using variations {variation_pairs}")

            if embedding_path is not None:
                print(f"Attempting to load embedding from {embedding_path}")
                self.diffusion_model.embedding_manager.load(
                    embedding_path, False
                )  # path, precision?

            out_direction = []
            if outpaint_direction is not None:
                assert (
                    init_img is not None
                ), "Must provide an initial image when using outpaint_direction"
                out_direction.append(outpaint_direction)
                if outpaint_pixels is not None:
                    out_direction.append(outpaint_pixels)

            init_img = str(init_img) if init_img else None
            init_mask = str(init_mask) if init_mask else None
            init_color = str(init_color) if init_color else None

            prediction_options = dict(
                prompt=prompt,
                init_img=init_img,
                init_mask=init_mask,
                init_color=init_color,
                out_direction=out_direction,
                strength=init_strength,
                fit=init_fit,
                variation_amount=variation_amount,
                with_variations=variation_pairs,
                iterations=num_images,
                steps=steps,
                width=width,
                height=height,
                sampler_name=sampler_name,
                cfg_scale=cfg_scale,
                seamless=seamless,
                facetool=facetool,
                gfpgan_strength=gfpgan_strength,
                codeformer_fidelity=codeformer_fidelity,
                embiggen=embiggen,
                embiggen_tiles=embiggen_tiles,
                upscale=upscale,
                seed=seed,
            )

        print(f"Running stable diffusion with prediction_options: {prediction_options}")
        prediction_options[
            "outdir"
        ] = self.current_outdir  # set the output directory regardless of the prompt
        generations = self.diffusion_model.prompt2png(**prediction_options)

        print(f"Generated {len(generations)} images")
        outputs = []
        for batch_index, (image_path, seed) in enumerate(generations):
            cog_output = ImageSeedOutput(image_path=Path(image_path), seed=seed)
            outputs.append(cog_output)
            print(f"image #{batch_index} - seed:{seed}")
        print(f"Generated {len(outputs)} images")
        return outputs
