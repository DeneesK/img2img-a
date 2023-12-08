# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
import random
sys.path.insert(0, "stylegan-encoder")
import tempfile  # noqa
from cog import BasePredictor, Input, Path  # noqa
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionControlNetPipeline
import torch  # noqa
from transformers import pipeline
import numpy as np

from diffusers.utils import load_image  # noqa


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False


def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make
        running multiple predictions efficient"""
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",
                                                     torch_dtype=torch.float16,
                                                     use_safetensors=True)
        self.pipeline = StableDiffusionControlNetPipeline.from_single_file(
            "https://huggingface.co/Timmek/anime_world/blob/main/anime_world_by_Timmek.safetensors",
            torch_dtype=torch.float16, use_safetensors=True,
            controlnet=controlnet
        )
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    def predict(
        self,
        image: Path = Input(description="input image"),
        prompt: str = Input(description="input prompt",
                            default='anime style'),
        negative_prompt: str = Input(description="input negative_prompt",
                                     default='easynegative, (bad-hands-5: 0.5)'),  # noqa
        seed: int = Input(description="input seed",
                          default=0),
        num_inference_steps: int = Input(
            description="input num_inference_steps",
            default=31
            ),
        guidance_scale: int = Input(
            description="input guidance_scale",
            default=7
        ),
        strength: float = Input(
            description="input strength",
            default=0.8
        )
    ) -> Path:
        """Run a single prediction on the model"""
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        try:
            image = load_image(str(image))
            depth_estimator = pipeline("depth-estimation")
            depth_map = get_depth_map(image, depth_estimator).unsqueeze(0).half().to("cuda")
            if not seed:
                seed = random.randint(0, 99999)
            generator = torch.Generator("cuda").manual_seed(seed)
            torch.cuda.empty_cache()
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            self.pipeline.safety_checker = disabled_safety_checker
            image = self.pipeline(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  image=image,
                                  control_image=depth_map,
                                  eta=1.0,
                                  generator=generator,
                                  num_inference_steps=int(num_inference_steps),
                                  guidance_scale=int(guidance_scale),
                                  strength=strength,
                                  ).images[0]
            image.save(out_path)
            return out_path
        except Exception as ex:
            print(ex)
