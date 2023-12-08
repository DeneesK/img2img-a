# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
import random
sys.path.insert(0, "stylegan-encoder")
import tempfile  # noqa
from cog import BasePredictor, Input, Path  # noqa
from diffusers import StableDiffusionImg2ImgPipeline
import torch  # noqa

from diffusers.utils import load_image  # noqa


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make
        running multiple predictions efficient"""
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self.pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
            "Timmek/anime_world",
            torch_dtype=torch.float16, use_safetensors=True
        )
        self.pipeline.enable_model_cpu_offload()
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
            init_image = load_image(str(image))
            if not seed:
                seed = random.randint(0, 99999)
            generator = torch.Generator("cuda").manual_seed(seed)
            torch.cuda.empty_cache()
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            self.pipeline.safety_checker = disabled_safety_checker
            image = self.pipeline(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  image=init_image,
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
