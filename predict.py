# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import sys
import random
sys.path.insert(0, "stylegan-encoder")
import tempfile  # noqa
from cog import BasePredictor, Input, Path  # noqa
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, LCMScheduler
import torch  # noqa
from controlnet_aux import OpenposeDetector, PidiNetDetector, HEDdetector, MediapipeFaceDetector

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
        # adapter_id = "latent-consistency/lcm-lora-sdv1-5"
        controlnet1 = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16
            )
        controlnet2 = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_scribble",
            torch_dtype=torch.float16
        )
        controlnet3 = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", subfolder="diffusion_sd15")
        controlnet = [controlnet1, controlnet2, controlnet3]
        self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            "dream.safetensors",
            torch_dtype=torch.float16, use_safetensors=True,
            controlnet=controlnet
        )
        # self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)
        # self.pipeline.load_lora_weights(adapter_id)
        # self.pipeline.fuse_lora()
        self.pipeline.enable_model_cpu_offload()
        print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    def predict(
        self,
        image: Path = Input(description="input image"),
        prompt: str = Input(description="input prompt",
                            default='A photo of a person, (anime style, colourful), cartoon'),
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
            default=0.7
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
        ),
        # low_threshold: int = Input(
        #     description="input FOR CANNY",
        #     default=100
        # ),
        # high_threshold: int = Input(
        #     description="input FOR CANNY",
        #     default=200
        # )
    ) -> Path:
        """Run a single prediction on the model"""
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        try:
            image = load_image(str(image))
            processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            processor2: PidiNetDetector = HEDdetector.from_pretrained('lllyasviel/Annotators')
            processor3: MediapipeFaceDetector = MediapipeFaceDetector()
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            control_image = processor(image, hand_and_face=True)
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            control_image2 = processor2(image, scribble=True)
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            control_image3 = processor3(image)
            if not seed:
                seed = random.randint(0, 99999)
            generator = torch.Generator("cuda").manual_seed(seed)
            torch.cuda.empty_cache()
            size = resize_(image)
            image = image.resize(size)
            control_image = control_image.resize(size)
            control_image2 = control_image2.resize(size)
            control_image3 = control_image3.resize(size)
            self.pipeline.safety_checker = disabled_safety_checker
            print('-------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            image = self.pipeline(prompt=prompt,
                                  negative_prompt=negative_prompt,
                                  image=image,
                                  control_image=[control_image,
                                                 control_image2,
                                                 control_image3],
                                  generator=generator,
                                  num_inference_steps=int(num_inference_steps),
                                  guidance_scale=int(guidance_scale),
                                  strength=strength,
                                  control_guidance_start=control_guidance_start,
                                  control_guidance_end=control_guidance_end,
                                  controlnet_conditioning_scale=controlnet_conditioning_scale
                                  ).images[0]
            make_image_grid([image,
                             control_image,
                             control_image2,
                             control_image3], rows=1, cols=4).save(out_path)
            return out_path
        except Exception as ex:
            print(ex)


def resize_(image) -> tuple[int, int]:
    w = image.width
    h = image.height

    if h < 1024 and w < 1024:
        if h % 8 == 0 and w % 8 == 0:
            return w, h
        w = w - (w % 8)
        h = h - (h % 8)
        return w, h

    while True:
        if h < 1024 and w < 1024:
            if h % 8 == 0 and w % 8 == 0:
                return w, h
            w = w - (w % 8)
            h = h - (h % 8)
            return w, h
        h = int(h / 2)
        w = int(w / 2)
