# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
import random
sys.path.insert(0, "stylegan-encoder")
import tempfile  # noqa
from cog import BasePredictor, Input, Path  # noqa
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, DPMSolverMultistepScheduler  # noqa
import torch  # noqa
from controlnet_aux import OpenposeDetector, HEDdetector  # noqa

from diffusers.utils import load_image, make_image_grid  # noqa


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
        controlnet1 = ControlNetModel.from_pretrained(
            "./control_v11p_sd15_openpose",
            torch_dtype=torch.float16
            )
        controlnet2 = ControlNetModel.from_pretrained(
            "./control_v11p_sd15_scribble",
            torch_dtype=torch.float16
        )
        controlnet = [controlnet1, controlnet2]
        self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            "dream.safetensors",
            torch_dtype=torch.float16, use_safetensors=True,
            controlnet=controlnet
        )
        self.processor = OpenposeDetector.from_pretrained('./ControlNet')
        self.processor2 = HEDdetector.from_pretrained('lllyasviel/Annotators')
        # self.pipeline.load_lora_weights('./', weight_name='animemix_v3_offset.safetensors')
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.enable_model_cpu_offload()
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    def predict(
        self,
        image: Path = Input(description="input image"),
        prompt: str = Input(description="input prompt",
                            default=''),
        negative_prompt: str = Input(description="input negative_prompt",
                                     default=''),
        seed: int = Input(description="input seed",
                          default=0,
                          ge=0,
                          le=10_000_000),
        num_inference_steps: int = Input(
            description="input num_inference_steps",
            default=61,
            ge=0,
            le=100
            ),
        guidance_scale: int = Input(
            description="input guidance_scale",
            default=6,
            ge=0,
            le=10
        ),
        strength: float = Input(
            description="input strength",
            default=0.55,
            ge=0,
            le=0.99
        ),
        controlnet_conditioning_scale: float = Input(
            description="""input controlnet_conditioning_scale, GENERAL,
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.""",
            default=0.8
        ),
        control_guidance_start: float = Input(
            description="""input control_guidance_start, GENERAL
            The percentage of total steps at which the ControlNet starts applying.
            """,
            default=0.0
        ),
        control_guidance_end: float = Input(
            description="input control_guidance_start, GENERAL. The percentage of total steps at which the ControlNet stops applying.",
            default=1.0
        )
    ) -> Path:
        """Run a single prediction on the model"""
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        try:
            image = load_image(str(image))

            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            control_image = self.processor(image, hand_and_face=True)
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            control_image2 = self.processor2(image, scribble=True)
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            if not seed:
                seed = random.randint(0, 99999)
            generator = torch.Generator("cuda").manual_seed(seed)
            torch.cuda.empty_cache()
            self.pipeline.safety_checker = disabled_safety_checker
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            w, h = resize_(image)
            print((w, h))
            image = self.pipeline(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  image=image,
                                  control_image=[control_image,
                                                 control_image2],
                                  generator=generator,
                                  num_inference_steps=int(num_inference_steps),
                                  guidance_scale=int(guidance_scale),
                                  strength=strength,
                                  control_guidance_start=control_guidance_start,
                                  control_guidance_end=control_guidance_end,
                                  controlnet_conditioning_scale=controlnet_conditioning_scale,
                                  width=w,
                                  height=h
                                  ).images[0]
            # image.save(out_path)
            # watermark_with_transparency(out_path)
            grid = make_image_grid(
                [image, control_image, control_image2],
                rows=1, cols=3
            )
            grid.save(grid)
            return out_path
        except Exception as ex:
            print(ex)


def resize_(image) -> tuple[int, int]:
    w = image.width
    h = image.height

    if w > h:
        c = h / w
        h = int(1024 * c)
        h = h - (h % 8)
        w = 1024
        return w, h

    c = w / h
    w = int(1024 * c)
    w = w - (w % 8)
    h = 1024
    return w, h

    # if h < 1024 and w < 1024:
    #     if h % 8 == 0 and w % 8 == 0:
    #         return w, h
    #     w = w - (w % 8)
    #     h = h - (h % 8)
    #     return w, h

    # while True:
    #     if h < 1024 and w < 1024:
    #         if h % 8 == 0 and w % 8 == 0:
    #             return w, h
    #         w = w - (w % 8)
    #         h = h - (h % 8)
    #         return w, h
    #     h = int(h / 2)
    #     w = int(w / 2)
