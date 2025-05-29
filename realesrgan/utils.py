import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # initialize model
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        # prefer to use params_ema
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def dni(self, net_a, net_b, dni_weight, key='params', loc='cpu'):
        """Deep network interpolation.

        ``Paper: Deep Network Interpolation for Continuous Imagery Effect Transition``
        """
        net_a = torch.load(net_a, map_location=torch.device(loc))
        net_b = torch.load(net_b, map_location=torch.device(loc))
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad for divisible borders
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        self.output = self.model(self.img)

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()
        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode
    
    # ========================= Batch Processing Methods =========================
    
    def pre_process_batch(self, imgs_np: list):
        """Pre-processes a batch of numpy images.
        Args:
            imgs_np (list[np.ndarray]): List of BGR numpy images of shape (H, W, C).
                                         All images are expected to be of the same size.
        Returns:
            torch.Tensor: Batch of pre-processed images as a tensor (N, C, H, W) on self.device.
                          Returns None if input list is empty.
        """
        if not imgs_np:
            return None

        batch_tensors = []
        # Assuming all images in the batch have the same dimensions and mod_scale requirements
        # We'll determine padding based on the first image, assuming it's representative.
        
        # Convert first image to determine padding requirements
        _img_for_padding_check = torch.from_numpy(np.transpose(imgs_np[0], (2, 0, 1))).float()
        
        # Pre-pad (if any) - this pad is applied before mod_pad
        # For batch processing, pre_pad is less common unless all images need exact same border handling
        # before model. We will apply pre_pad here if self.pre_pad != 0
        # However, if pre_pad is large, it might be better handled by individual image padding before batching
        # For simplicity, we assume pre_pad is 0 or small enough for uniform batch application.
        # If self.pre_pad != 0, a more robust solution might be needed if images vary a lot.
        # Let's keep the original logic's spirit: pre_pad is applied, then mod_pad.

        current_mod_pad_h, current_mod_pad_w = 0, 0
        if self.scale == 2:
            current_mod_scale = 2
        elif self.scale == 1: # Typically for RealESRGAN, scale is 2 or 4. Scale 1 might be for other variants.
            current_mod_scale = 4
        else: # For scale 4, or others, often no mod_scale is enforced by default in some versions,
              # or it's implicitly handled by network architecture or specific model needs.
              # We'll follow the existing mod_scale logic. If self.scale is 4, self.mod_scale remains None from init.
            current_mod_scale = self.mod_scale # Use self.mod_scale determined in __init__ or pre_process for single image

        if current_mod_scale is not None:
            # Temporarily use _img_for_padding_check to calculate mod_pad
            # Apply pre_pad first if it exists
            if self.pre_pad != 0:
                _img_for_padding_check = F.pad(_img_for_padding_check.unsqueeze(0), (0, self.pre_pad, 0, self.pre_pad), 'reflect').squeeze(0)

            _, h_temp, w_temp = _img_for_padding_check.size() # Get H, W after potential pre_pad
            if h_temp % current_mod_scale != 0:
                current_mod_pad_h = current_mod_scale - h_temp % current_mod_scale
            if w_temp % current_mod_scale != 0:
                current_mod_pad_w = current_mod_scale - w_temp % current_mod_scale
        
        # Store these calculated paddings to be used in post_process_batch
        self.batch_mod_pad_h = current_mod_pad_h 
        self.batch_mod_pad_w = current_mod_pad_w

        for img_np in imgs_np:
            img_tensor = torch.from_numpy(np.transpose(img_np, (2, 0, 1))).float().to(self.device)
            if self.half:
                img_tensor = img_tensor.half()

            if self.pre_pad != 0:
                img_tensor = F.pad(img_tensor.unsqueeze(0), (0, self.pre_pad, 0, self.pre_pad), 'reflect').squeeze(0)
            
            if current_mod_scale is not None and (current_mod_pad_h > 0 or current_mod_pad_w > 0) : # Only pad if necessary
                img_tensor = F.pad(img_tensor.unsqueeze(0), (0, self.batch_mod_pad_w, 0, self.batch_mod_pad_h), 'reflect').squeeze(0)
            
            batch_tensors.append(img_tensor)
        
        self.batch_imgs_tensor = torch.stack(batch_tensors, dim=0)
        return self.batch_imgs_tensor

    def process_batch(self):
        """Processes the batch of images using the model."""
        # Model inference on the entire batch
        self.batch_output_tensor = self.model(self.batch_imgs_tensor)

    def post_process_batch(self):
        """Post-processes the batch of output images.
        Removes padding and converts tensors back to a list of numpy images.
        """
        # Retrieve mod_pad values calculated in pre_process_batch
        current_mod_pad_h = getattr(self, 'batch_mod_pad_h', 0)
        current_mod_pad_w = getattr(self, 'batch_mod_pad_w', 0)
        
        # Determine current_mod_scale based on self.scale, similar to pre_process
        current_mod_scale = None
        if self.scale == 2:
            current_mod_scale = 2
        elif self.scale == 1:
            current_mod_scale = 4
        # Add other scale conditions if necessary, or rely on self.mod_scale if set for other scales

        # Remove extra mod pad
        if current_mod_scale is not None and (current_mod_pad_h > 0 or current_mod_pad_w > 0):
            _, _, h, w = self.batch_output_tensor.size()
            self.batch_output_tensor = self.batch_output_tensor[:, :, 0:h - current_mod_pad_h * self.scale, 0:w - current_mod_pad_w * self.scale]
        
        # Remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.batch_output_tensor.size()
            # Ensure pre_pad * self.scale doesn't exceed dimensions
            h_unpad = max(0, h - self.pre_pad * self.scale)
            w_unpad = max(0, w - self.pre_pad * self.scale)
            self.batch_output_tensor = self.batch_output_tensor[:, :, 0:h_unpad, 0:w_unpad]
        
        # Convert batch tensor to list of numpy images
        output_imgs_np = []
        for i in range(self.batch_output_tensor.size(0)):
            output_img_tensor = self.batch_output_tensor[i].squeeze().float().cpu().clamp_(0, 1).numpy()
            output_img_np = np.transpose(output_img_tensor[[2, 1, 0], :, :], (1, 2, 0)) # RGB to BGR
            output_imgs_np.append(output_img_np)
            
        return output_imgs_np

    @torch.no_grad()
    def enhance_batch_true(self, imgs_list: list, outscale=None):
        """Upsamples a batch of images using RealESRGAN with true model-level batching.
        Assumes tile_size = 0 (no tiling for batch mode).
        Args:
            imgs_list (list[np.ndarray]): A list of BGR numpy images.
                                          All images MUST be of the same shape (H, W, C)
                                          and normalized to [0, 1] range, float32 type.
            outscale (float): The final output scale of the image. Default: None.
        Returns:
            list[np.ndarray]: List of upsampled BGR numpy images (uint8).
        """
        if not imgs_list:
            return []
        
        # Store original input dimensions for potential final resizing by outscale
        h_input, w_input = imgs_list[0].shape[0:2] 

        # Normalize images to [0, 1] range (input to this function should already be BGR, float32, 0-1)
        # The original enhance method does normalization. We need to ensure inputs to this batch method are consistent.
        # For simplicity, we assume the input `imgs_list` are already BGR, np.float32, normalized to [0,1].
        # If they are not, normalization should happen before calling this function or at the beginning here.
        
        # Example: If imgs_list contains uint8 BGR images:
        processed_imgs_list = []
        for img_np_uint8 in imgs_list:
            img_np_float32 = img_np_uint8.astype(np.float32) / 255.
            # Assuming BGR input, convert to RGB for pre_process_batch as it expects RGB transposition
            img_rgb_float32 = cv2.cvtColor(img_np_float32, cv2.COLOR_BGR2RGB)
            processed_imgs_list.append(img_rgb_float32)

        # --- Pre-process the batch of images ---
        # self.pre_process_batch expects a list of RGB float32 numpy images
        batch_input_tensor = self.pre_process_batch(processed_imgs_list)
        if batch_input_tensor is None:
            return []

        # --- Process the batch using the model ---
        if self.tile_size > 0:
            # Batch processing with tiling is complex and not implemented here.
            # Fallback or raise error. For now, we assume tile_size = 0 for this batch method.
            print("Warning: enhance_batch_true does not support tiling. Process might be slow or OOM for large images in batch.")
            # Potentially, one could loop through the batch and apply single-image enhance with tiling:
            # return [self.enhance(img, outscale=outscale)[0] for img in imgs_list] # This defeats batch purpose
            # Or simply proceed without tiling, risking OOM for very large images in the batch.
            # For now, let's assume self.tile_size is set to 0 if this method is called.
            # If self.tile_size was intended, the caller should iterate and use self.enhance().
            self.process_batch() # self.batch_imgs_tensor was set in pre_process_batch
        else:
            self.process_batch() # self.batch_imgs_tensor was set in pre_process_batch
        
        # --- Post-process the batch ---
        output_imgs_rgb_np_list = self.post_process_batch() # Returns list of RGB float32 numpy images

        final_output_list = []
        for output_img_rgb_np in output_imgs_rgb_np_list:
            # Convert RGB back to BGR
            output_img_bgr_np = cv2.cvtColor(output_img_rgb_np, cv2.COLOR_RGB2BGR)

            # Denormalize from [0,1] to [0,255] and convert to uint8
            # Assuming max_range is 255 as 16-bit image batching is not explicitly handled here.
            output_uint8 = (output_img_bgr_np * 255.0).round().astype(np.uint8)

            if outscale is not None and outscale != float(self.scale):
                output_uint8 = cv2.resize(
                    output_uint8,
                    (int(w_input * outscale), int(h_input * outscale)),
                    interpolation=cv2.INTER_LANCZOS4
                )
            final_output_list.append(output_uint8)
            
        return final_output_list


class PrefetchReader(threading.Thread):
    """Prefetch images.

    Args:
        img_list (list[str]): A image list of image paths to be read.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)

        self.que.put(None)

    def __next__(self):
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class IOConsumer(threading.Thread):

    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break

            output = msg['output']
            save_path = msg['save_path']
            cv2.imwrite(save_path, output)
        print(f'IO worker {self.qid} is done.')
