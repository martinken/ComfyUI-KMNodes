import kornia
import nodes
import node_helpers
import torch
from torch import Tensor
import cv2
import numpy as np
import comfy.model_management
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from scipy.optimize import least_squares # for color matching
from skimage.transform import resize # for downscale image

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)

class KM_Safe_Mask_Bounds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KM_Safe_Mask_Bounds",
            category="KMNodes",
            description=
                "Compute a safe bounding box for an image given a mask and a factor to grow by. "
                "This is useful for workflows such as where florance2 is generating a mask but "
                "you want to safely expand it by a factor without going beyond the image bounds.",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("x", default = 0, min = 0, max = 4096, tooltip = "x position of mask."),
                io.Int.Input("y", default = 0, min = 0, max = 4096, tooltip = "y position of mask."),
                io.Int.Input("mask_width", default = 0, min = 0, max = 4096, tooltip = "width of mask."),
                io.Int.Input("mask_height", default = 0, min = 0, max = 4096, tooltip = "height of mask."),
                io.Float.Input("grow", default = 0.5, min = 0, max = 2.0, tooltip = "Amount (as a ratio) to grow the bounds by."),
                io.Float.Input("aspect", default = 1.0, min = 0.1, max = 10.0, tooltip = "desired aspect ratio X / Y."),
            ],
            outputs=[
                io.Int.Output(display_name="x"),
                io.Int.Output(display_name="y"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, image, x, y, mask_width, mask_height, grow, aspect) -> io.NodeOutput:
        xfactor = 1.0
        yfactor = 1.0
        if aspect > 1.0:
            xfactor = aspect
        if aspect < 1.0:
            yfactor = 1.0/aspect
        target = max(mask_width*xfactor, mask_height*yfactor)
        target = target * (1.0 + grow) * 0.5
        mid_x = x + mask_width * 0.5
        mid_y = y + mask_height * 0.5
        if mid_x - target*xfactor < 0:
            target = mid_x/xfactor
        if mid_y - target*yfactor < 0:
            target = mid_y/yfactor
        if mid_x + target*xfactor >= image.shape[2]:
            target = (image.shape[2] - mid_x)/xfactor
        if mid_y + target*yfactor >= image.shape[1]:
            target = (image.shape[1] - mid_y)/yfactor

        return (int(mid_x - target*xfactor), int(mid_y - target*yfactor), int(target*2*xfactor), int(target*2*yfactor))

class KM_Safe_SEGS_Bounds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KM_Safe_SEGS_Bounds",
            category="KMNodes",
            description=
                "Compute a safe bounding box for an image given a SEGS and a factor to grow by. "
                "This is useful for workflows such as where florance2 is generating a mask but "
                "you want to safely expand it by a factor without going beyond the image bounds.",
            inputs=[
                io.Image.Input("image"),
                io.SEGS.Input("segs"),
                io.Float.Input("grow", default=0.5, min=0, max=2.0, tooltip="Amount (as a ratio) to grow the bounds by."),
            ],
            outputs=[
                io.Int.Output(display_name="x"),
                io.Int.Output(display_name="y"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, image, segs, grow) -> io.NodeOutput:
        for seg in segs[1]:
            crop_region = seg.crop_region

        crop_region = seg.crop_region

        width = crop_region[3] - crop_region[1] + 1
        height = crop_region[2] - crop_region[0] + 1

        target = max(width, height)
        target = target * (1.0 + grow) * 0.5
        mid_x = crop_region[1] + width * 0.5
        mid_y = crop_region[0] + height * 0.5
        if mid_x - target < 0:
            target = mid_x
        if mid_y - target < 0:
            target = mid_y
        if mid_x + target >= image.shape[2]:
            target = image.shape[2] - mid_x
        if mid_y + target >= image.shape[1]:
            target = image.shape[1] - mid_y

        return (int(mid_x - target), int(mid_y - target), int(target*2), int(target*2))

#comfy really needs fist class support for looping
class KM_Merge_Images(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KM_Merge_Images",
            category="KMNodes",
            description=
                "Merge two sets of images with support for Video Helper Suite meta_batch manager. "
                "I forget if this works or not as I have not used it in a bit. ",
            inputs=[
                io.Image.Input("images_A"),
                io.Image.Input("images_B"),
                io.AnyType.Input("meta_batch", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="merged_image"),
            ],
            hidden=[
                io.Hidden.unique_id,
            ]
        )

    @classmethod
    def execute(cls, images_A: Tensor, images_B: Tensor, meta_batch = None, unique_id = None) -> io.NodeOutput:
        # handle non batched case
        if meta_batch is None:
            images = []
            images.append(images_A)
            images.append(images_B)
            all_images = torch.cat(images, dim=0)
            return (all_images,)

        # if here, then we have a meta_batch
        # for inp in prompt[unique_id]['inputs'].values():
                
                # for output_uid in prompt:
                #     if prompt[output_uid]['class_type'] in ["VHS_VideoCombine"]:
                #         for inp in prompt[output_uid]['inputs'].values():
                #             if inp == [bm_uid, 0]:
                #                 managed_outputs+=1


        if unique_id not in meta_batch.inputs:
            frames_done = 0
            meta_batch.inputs[unique_id] = (frames_done,)
        else:
            frames_done = meta_batch.inputs[unique_id]

        # print(f"""\033[96mTotal: {meta_batch.total_frames} 
        #       done: {frames_done}
        #       shape: {images_A.shape}
        #       batch: {meta_batch}
        #       id: {unique_id}\033[0m""")

        images = []
        images.append(images_A)
        frames_done += images_A.shape[0]
        meta_batch.inputs[unique_id] = (frames_done)

        if frames_done  >= meta_batch.total_frames:            
            meta_batch.inputs.pop(unique_id)
            images.append(images_B)

        all_images = torch.cat(images, dim=0)
        return (all_images,)


class KM_Aspect_Ratio_Selector(io.ComfyNode):
    RATIO = [
        ("1:1  1M 960x960", 960, 960),
        ("4:3  1M 1088x816", 1088, 816),
        ("16:9 1M 1280x720", 1280, 720),
        ("3:4  1M 816x1088", 816, 1088),
        ("9:16 1M 720x1280", 720, 1280),
        ("4:3      960x720", 960, 720),
        ("3:4      720x960", 720, 920),
        ("1:1  2M 1440x1440", 1440, 1440),
        ("4:3  2M 1632x1440", 1632, 1224),
        ("3:2  2M 1752x1168", 1752, 1168),
        ("16:9 2M 1920x1080", 1920, 1080),
        ("3:4  2M 1440x1632", 1224, 1632),
        ("2:3  2M 1168x1752", 1168, 1752),
        ("9:16 2M 1080x1920", 1080, 1920),
        ("custom", 960, 960),
    ]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KM_Aspect_Ratio_Selector",
            category="KMNodes",
            inputs=[
                io.Combo.Input("aspect_ratio",
                    options=[title for title, _, _ in KM_Aspect_Ratio_Selector.RATIO],
                    default=KM_Aspect_Ratio_Selector.RATIO[0][0],
                    tooltip="Aspect ratio of generated image.",
                ),
                io.Int.Input("width", default=512, min=0, max=4096, step=1, optional=True),
                io.Int.Input("height", default=512, min=0, max=4096, step=1, optional=True),
            ],
            outputs=[
                io.String.Output(display_name="ratio"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ]
        )

    @classmethod
    def execute(cls, aspect_ratio, custom_width = 512, custom_height = 512) -> io.NodeOutput:
        if aspect_ratio != "custom":
            for title, w, h in cls.RATIO:
                if title == aspect_ratio:
                    return (title, w, h)
        return (aspect_ratio, custom_width, custom_height)

class KM_Video_Image_Color_Match(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KM_Video_Image_Color_Match",
            category="KMNodes",
            description=
                "Make a set of images match the provided reference images. "
                "This is useful for workflows such as extending a video using Wan "
                "where there is a color shift in the appended video. This node works "
                "by computing image properties on the reference images and then "
                "adjusting the images to match. Given that Wan tends to impact gamma "
                "as well as luminance the LS-LS mode is suggested. It uses a least "
                "squares technique to compute a mapping from reference to input "
                "for the function y' = a*y^b+c for the L and S channels in HLS color "
                "space. The reference_location should be set to start or end depending "
                "on if you are appending video or prepending video.",
            inputs=[
                io.Image.Input("images", 
                    tooltip="Images generated by the AI model that need to be color matched."),
                io.Image.Input("reference",
                    tooltip="Images from a previously generated video (overlap FF or LF) you are trying to match."),
                io.Float.Input("factor", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="How much correction to apply"),
                io.Boolean.Input("blend", default=True,
                    tooltip="If true, slowly blend between the reference images to the AI Images for the frames that overlap."),
                io.Combo.Input("reference_location", options=["start", "end"],
                    tooltip="are the reference frames at the start or end of the generated AI frames"),
                io.Combo.Input("color_space", options=["HLS", "LAB", "Linear", "RGB", "LS-LS"], default="LS-LS",
                    tooltip="Color space to use for matching"),
                io.Combo.Input("device", options=["auto", "cpu", "gpu"],
                    tooltip="Device to use for computation"),
            ],
            outputs=[
                io.Image.Output(display_name="result"),
            ]   
        )

    @classmethod
    def execute(cls, images, reference, factor, blend, reference_location, color_space, device) -> io.NodeOutput:
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        # Ensure image and reference are in the correct shape (B, C, H, W)
        reference = reference.permute([0, 3, 1, 2]).to(device)
        images = images.permute([0, 3, 1, 2]).to(device)

        match_index = reference.shape[0] - 1
        if reference_location == "start":
            color_stats = cls.analyze_color_statistics(reference[match_index:match_index+1], 
                                                       images[match_index:match_index+1], color_space)
        else:
            color_stats = cls.analyze_color_statistics(reference[-1:], images[-1:], color_space)

        # Apply color transformation
        result = cls.apply_color_transformation(
            images, color_stats, color_space
            )

        # Apply factor
        result = factor * result + (1 - factor) * images

        # blend
        num_refs = reference.shape[0]
        if blend:
            for i in range(num_refs):
                blend_factor = (i+1.0)/(num_refs + 1.0)
                if reference_location == "start":
                    result[i] = blend_factor*result[i] + (1.0 - blend_factor)*reference[i]
                else:
                    j = result.shape[0] - reference.shape[0] + i
                    result[j] = (1.0 - blend_factor)*result[j] + blend_factor*reference[i]

        # Convert back to (B, H, W, C) format and ensure values are in [0, 1] range
        result = result.permute(0, 2, 3, 1).clamp(0, 1).to(comfy.model_management.intermediate_device())

        return (result,)

    @classmethod
    def mapping_function(cls, a, x):
        return a[0]*np.power(x,a[1]) + a[2]

    @classmethod
    def error_function(cls, a, x, y):
        return cls.mapping_function(a, x) - y

    @classmethod
    # compute mapping from dest_image to image
    def analyze_color_statistics(cls, image, dest_image, color_space):
        # Assuming image is in RGB format
        if "LAB" == color_space:
            image = kornia.color.rgb_to_lab(image)
            dest_image = kornia.color.rgb_to_lab(dest_image)
        elif "HLS" == color_space or "LS-LS" == color_space:
            image = kornia.color.rgb_to_hls(image)
            dest_image = kornia.color.rgb_to_hls(dest_image)
        elif "Linear" == color_space:
            image = kornia.color.rgb_to_linear_rgb(image)
            dest_image = kornia.color.rgb_to_linear_rgb(dest_image)

        result = {}
        i1, i2, i3 = image.chunk(3, dim=1)
        d1, d2, d3 = dest_image.chunk(3, dim=1)

        if color_space == "LS-LS":
            d2 = d2.flatten()
            d3 = d3.flatten()
            i2 = i2.flatten()
            i3 = i3.flatten()
            a0 = [1.0, 1.0, 0.0] #initial guess
            result["fit_c2"] = least_squares(cls.error_function, a0, args = (d2,i2))
            a0 = [1.0, 1.0, 0.0] #initial guess
            result["fit_c3"] = least_squares(cls.error_function, a0, args = (d3,i3))
        else:
            result["scale_c1"] = i1.std()/d1.std()
            result["shift_c1"] = i1.mean() - d1.mean()*result["scale_c1"]
            result["scale_c2"] = i2.std()/d2.std()
            result["shift_c2"] = i2.mean() - d2.mean()*result["scale_c2"]
            result["scale_c3"] = i3.std()/d3.std()
            result["shift_c3"] = i3.mean() - d3.mean()*result["scale_c3"]

        print(f"""\033[96mrf: {result}
              \033[0m""")
        return result

    @classmethod
    def apply_color_transformation(cls, image, color_transform, color_space):
        if "LAB" == color_space:
            image = kornia.color.rgb_to_lab(image)
            v1, v2, v3 = image.chunk(3, dim=1)
            v1_new = v1*color_transform["scale_c1"] + color_transform["shift_c1"]
            v2_new = v2*color_transform["scale_c2"] + color_transform["shift_c2"]
            v3_new = v3*color_transform["scale_c3"] + color_transform["shift_c3"]
        elif "HLS" == color_space:
            image = kornia.color.rgb_to_hls(image)
            v1, v2, v3 = image.chunk(3, dim=1)
            v1_new = v1
            v2_new = v2*color_transform["scale_c2"] + color_transform["shift_c2"]
            v3_new = v3*color_transform["scale_c3"] + color_transform["shift_c3"]
        elif "Linear" == color_space:
            image = kornia.color.rgb_to_linear_rgb(image)
            v1, v2, v3 = image.chunk(3, dim=1)
            v1_new = v1*color_transform["scale_c1"] + color_transform["shift_c1"]
            v2_new = v2*color_transform["scale_c2"] + color_transform["shift_c2"]
            v3_new = v3*color_transform["scale_c3"] + color_transform["shift_c3"]
        elif "LS-LS" == color_space:
            print(f"""\033[96mrf: {color_transform}
                  \033[0m""")
            image = kornia.color.rgb_to_hls(image)
            v1, v2, v3 = image.chunk(3, dim=1)
            v1_new = v1
            v2_new = color_transform["fit_c2"].x[0]*pow(v2,color_transform["fit_c2"].x[1]) + color_transform["fit_c2"].x[2]
            v3_new = color_transform["fit_c3"].x[0]*pow(v3,color_transform["fit_c3"].x[1]) + color_transform["fit_c3"].x[2]
        else:
            v1, v2, v3 = image.chunk(3, dim=1)
            v1_new = v1*color_transform["scale_c1"] + color_transform["shift_c1"]
            v2_new = v2*color_transform["scale_c2"] + color_transform["shift_c2"]
            v3_new = v3*color_transform["scale_c3"] + color_transform["shift_c3"]

        # Combine channels
        rgb_new = torch.cat([v1_new, v2_new, v3_new], dim=1)

        # Convert back to RGB
        if "LAB" == color_space:
            rgb_new = kornia.color.lab_to_rgb(rgb_new)
        elif "HLS" == color_space or "LS-LS" == color_space:
            rgb_new = kornia.color.hls_to_rgb(rgb_new)
        elif "Linear" == color_space:
            rgb_new = kornia.color.linear_rgb_to_rgb(rgb_new)

        return rgb_new


class KM_Color_Correct(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KM_Color_Correct",
            category="KMNodes",
            description=
                "A simple image color adjustment node. ",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("luminance", default=1.0, min=0.5, max=1.5, step=0.01),
                io.Float.Input("saturation", default=1.0, min=0.0, max=1.5, step=0.01),
                io.Float.Input("gamma", default=1.0, min=0.2, max=2.2, step=0.1),
                io.Float.Input("red_gain", default=1.0, min=0.7, max=1.3, step=0.01),
                io.Float.Input("green_gain", default=1.0, min=0.7, max=1.3, step=0.01),
                io.Float.Input("blue_gain", default=1.0, min=0.7, max=1.3, step=0.01),
                io.Float.Input("red_shift", default=0.0, min=-0.2, max=0.2, step=0.01),
                io.Float.Input("green_shift", default=0.0, min=-0.2, max=0.2, step=0.01),
                io.Float.Input("blue_shift", default=0.0, min=-0.2, max=0.2, step=0.01),
            ],
            outputs=[
                io.Image.Output(display_name="result"),
            ]
        )

    @classmethod
    def execute(
        self,
        image: torch.Tensor,
        luminance: float,
        saturation: float,
        gamma: float,
        red_gain: float,
        green_gain: float,
        blue_gain: float,
        red_shift: float,
        green_shift: float,
        blue_shift: float
    ) -> io.NodeOutput:
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            modified_image = image[b].numpy().astype(np.float32)

            # gamma
            modified_image = np.clip(np.power(modified_image, gamma), 0, 1)

            # luminance and saturation
            hls_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HLS)
            hls_img[:, :, 1] = np.clip(luminance * hls_img[:, :, 1], 0, 1)
            hls_img[:, :, 2] = np.clip(saturation * hls_img[:, :, 2], 0, 1)
            modified_image = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB)

            # gains
            modified_image[:, :, 0] *= red_gain
            modified_image[:, :, 1] *= green_gain
            modified_image[:, :, 2] *= blue_gain

            modified_image[:, :, 0] += red_shift
            modified_image[:, :, 1] += green_shift
            modified_image[:, :, 2] += blue_shift

            modified_image = torch.from_numpy(modified_image).unsqueeze(0)
            result[b] = modified_image

        return (result,)


class KM_WanImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanImageToVideo",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None) -> io.NodeOutput:
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent = torch.zeros([batch_size, latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())
        if latent_channels == 48:
            concat_latent = comfy.latent_formats.Wan22().process_out(concat_latent)
        else:
            concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
        concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(start_image[:, :, :, :3])
            concat_latent[:,latent_channels:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]
            mask[:, :, :start_image.shape[0] + 3] = 0.0

            # start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            # image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            # image[:start_image.shape[0]] = start_image

            # concat_latent_image = vae.encode(image[:, :, :, :3])
            # mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            # mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent, "concat_mask": mask, "concat_mask_index": latent_channels})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent, "concat_mask": mask, "concat_mask_index": latent_channels})
            # positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            # negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)
    
class KM_Downscale_Image(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KM_Downscale_Image",
            category="KMNodes",
            description=
                "Downscale an image properly to avoid aliasing artifacts. Uses skimage "
                "which does a proper blur before downsampling. ",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("width", default=128, min=0, max=BIGMAX, step=1),
                io.Int.Input("height", default=128, min=0, max=BIGMAX, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="result"),
            ]   
        )

    @classmethod
    def execute(cls, images, width, height) -> io.NodeOutput:
        # short circuit
        if (width == images.size(2)) and (height == images.size(1)):
            return (images,)
        
        # Ensure image and reference are in the correct shape (B, H, W, C)
        result = torch.empty((images.size(0), height, width, images.size(3)), dtype=images.dtype, layout=images.layout)
        for i, image in enumerate(images):
            image_resized = resize(image, (height, width), anti_aliasing=True)
            result[i] = torch.from_numpy(image_resized).unsqueeze(0) 

        return (result,)


class KM_WanVideoToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KM_WanVideoToVideo",
            category="KMNodes",
            description=
                "Take an input video and encode it into a latent. This can then be used "
                "as input to a sampler as an init video similar to image workflows that use "
                "init images. For example you could take a video generated with Wan and "
                "and crop a region of it, zoom it, then use this node to feed it back "
                "into Wan t2v to refine the cropped region or do more "
                "creative options. Basically all the same operations you would do with an "
                "init image when using Flux for example. ",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("video"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, video=None) -> io.NodeOutput:
        # spacial_scale = vae.spacial_compression_encode()
        # latent = torch.zeros([batch_size, vae.latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale], device=comfy.model_management.intermediate_device())
        video = comfy.utils.common_upscale(video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        latent = vae.encode(video[:, :, :, :3])

        # image = torch.ones((length, height, width, 3)) * 0.5
        # mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        # if video is not None:
        #     image[:video.shape[0]] = video
        #     mask[:, :, :video.shape[0] + 3] = mask_strength

        # concat_latent_image = vae.encode(image[:, :, :, :3])
        # mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        # positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        # negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        # clip_vision_output = None
        # if clip_vision_start_image is not None:
        #     clip_vision_output = clip_vision_start_image

        # if clip_vision_output is not None:
        #     positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
        #     negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return io.NodeOutput(positive, negative, out_latent)

class KMNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            KM_WanVideoToVideo,
            KM_Downscale_Image,
            KM_WanImageToVideo,
            KM_Safe_Mask_Bounds,
            KM_Safe_SEGS_Bounds,
            KM_Merge_Images,
            KM_Aspect_Ratio_Selector,
            KM_Video_Image_Color_Match,
            KM_Color_Correct,
        ]

async def comfy_entrypoint() -> KMNodesExtension:
    return KMNodesExtension()
