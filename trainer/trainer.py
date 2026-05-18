import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F

from utils import Logger

from lpips import LPIPS
from jaxtyping import Float, UInt8
from skimage.metrics import structural_similarity
from einops import reduce

from PIL import Image
from pathlib import Path
from einops import rearrange, repeat
from typing import Union
import numpy as np
import heapq
import json

from tqdm import tqdm

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]

class DrivingForwardTrainer:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank, use_tb=True):
        self.read_config(cfg)
        self.rank = rank        
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
            self.depth_metric_names = self.logger.get_metric_names()

        self.lpips = LPIPS(net="vgg").cuda(rank)

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def learn(self, model):
        """
        This function sets training process.
        """        
        train_dataloader = model.train_dataloader()
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        start_time = time.time()
        for self.epoch in range(self.num_epochs):
                
            self.train(model, train_dataloader, start_time)
            
            # save model after each epoch using rank 0 gpu 
            if self.rank == 0:
                model.save_model(self.epoch)
                print('-'*110) 
                
            if self.ddp_enable:
                dist.barrier()
                
        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        This function trains models.
        """
        model.set_train()
        pbar = tqdm(total=len(data_loader), desc='training on epoch {}'.format(self.epoch), mininterval=100)
        for batch_idx, inputs in enumerate(data_loader):         
            before_op_time = time.time()
            model.optimizer.zero_grad(set_to_none=True)
            outputs, losses = model.process_batch(inputs, self.rank)
            losses['total_loss'].backward()
            model.optimizer.step()

            if self.rank == 0: 
                self.logger.update(
                    'train', 
                    self.epoch, 
                    self.world_size,
                    batch_idx, 
                    self.step,
                    start_time,
                    before_op_time, 
                    inputs,
                    outputs,
                    losses
                )

                if self.logger.is_checkpoint(self.step):
                    self.validate(model)

            self.step += 1
            pbar.update(1)

            # for n, p in model.named_parameters():
            #     if 'pose_mlp' in n:
            #         print(n, p.grad is None, p.grad.abs().mean().item() if p.grad is not None else None)
            #     if 'encoder' in n:
            #         print(n, p.grad is None, p.grad.abs().mean().item() if p.grad is not None else None)
        pbar.close()
        model.lr_scheduler.step()
        
    @torch.no_grad()
    def validate(self, model, vis_results=False):
        """
        This function validates models on the validation dataset to monitor training process.
        """
        val_dataloader = model.val_dataloader()
        val_iter = iter(val_dataloader)
        
        # Ensure the model is in validation mode
        model.set_val()

        avg_reconstruction_metric = defaultdict(float)

        inputs = next(val_iter)
        outputs, _ = model.process_batch(inputs, self.rank)
            
        psnr, ssim, lpips, gaussian_num, render_time = self.compute_reconstruction_metrics(inputs, outputs)

        avg_reconstruction_metric['psnr'] += psnr   
        avg_reconstruction_metric['ssim'] += ssim
        avg_reconstruction_metric['lpips'] += lpips
        avg_reconstruction_metric['gaussian_num'] += gaussian_num
        avg_reconstruction_metric['render_time'] += render_time

        print('Validation reconstruction result...\n')
        print(f"\n{inputs['token'][0]}")
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')

        # Set the model back to training mode
        model.set_train()

    @torch.no_grad()
    def evaluate(self, model, scale = 0):
        """
        This function evaluates models on validation dataset of samples with context.
        """
        eval_dataloader = model.eval_dataloader()

        # load model
        model.load_weights()
        model.set_eval()

        avg_reconstruction_metric = defaultdict(float)

        count = 0

        # Keep top-k scenes by PSNR (min-heap of size k)
        top_k = 2
        top_scenes = []  # entries: (psnr_value, token, inputs, outputs)

        process = tqdm(eval_dataloader)
        false_cnt = 0
        for batch_idx, inputs in enumerate(process):
            outputs, _ = model.process_batch(inputs, self.rank)
            
            psnr, ssim, lpips, gaussian_num, render_time = self.compute_reconstruction_metrics(inputs, outputs, scale=scale)

            if count > 30 and render_time > 0.1:
                render_time = 0
                false_cnt += 1
            # accumulate averages
            avg_reconstruction_metric['psnr'] += psnr   
            avg_reconstruction_metric['ssim'] += ssim
            avg_reconstruction_metric['lpips'] += lpips
            avg_reconstruction_metric['gaussian_num'] += gaussian_num
            avg_reconstruction_metric['render_time'] += render_time
            count += 1

            psnr_value = float(psnr)
            lpips_value = float(lpips)
            ssim_value = float(ssim)

            # maintain top-k scenes
            token = inputs['token'][0]

            heapq.heappush(top_scenes, (ssim_value, token, inputs, outputs))
            if len(top_scenes) > top_k:
                heapq.heappop(top_scenes)

            process.set_description(f"PSNR: {psnr_value:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}, Gaussian Num: {gaussian_num:.2f}, Render Time: {render_time:.8f}s")

            print(f"\n{token}")
            print(f"avg PSNR: {avg_reconstruction_metric['psnr']/count:.4f}, avg SSIM: {avg_reconstruction_metric['ssim']/count:.4f}, avg LPIPS: {avg_reconstruction_metric['lpips']/count:.4f}, avg Gaussian Num: {avg_reconstruction_metric['gaussian_num']/count:.2f}, avg Render Time: {avg_reconstruction_metric['render_time']/count:.8f}s")
            
        avg_reconstruction_metric['psnr'] /= len(eval_dataloader)
        avg_reconstruction_metric['ssim'] /= len(eval_dataloader)
        avg_reconstruction_metric['lpips'] /= len(eval_dataloader)
        avg_reconstruction_metric['gaussian_num'] /= len(eval_dataloader)
        avg_reconstruction_metric['render_time'] /= (len(eval_dataloader) - false_cnt)

        print(f"false render count: {false_cnt}")


        print('Evaluation reconstruction result...\n')
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')

        # Save top-k scenes outputs into test_images directory
        test_images_dir = Path('test_images/ssim')
        if len(top_scenes) > 0:
            # sort descending by PSNR
            top_sorted = sorted(top_scenes, key=lambda x: x[0], reverse=True)
            for rank_idx, (psnr_value, token, inputs, outputs) in enumerate(top_sorted, start=1):
                if self.novel_view_mode == 'SF':
                    frame_id = 1
                elif self.novel_view_mode == 'MF':
                    frame_id = 0
                else:
                    raise ValueError(f"Invalid novel view mode: {self.novel_view_mode}")
                                # Save extrinsics for this scene (if present)
                extrinsics = inputs.get('extrinsics', None)
                if extrinsics is not None:
                    try:
                        if hasattr(extrinsics, 'detach'):
                            ex_list = extrinsics.detach().cpu().numpy().tolist()
                        else:
                            ex_list = np.array(extrinsics).tolist()
                        dest = test_images_dir / token
                        dest.mkdir(parents=True, exist_ok=True)
                        with open(dest / 'extrinsics.json', 'w') as f:
                            json.dump(ex_list, f)
                    except Exception as e:
                        print(f"Failed to save extrinsics for {token}: {e}")
                for cam in range(self.num_cams):
                    rgb_gt = inputs[('color', frame_id, 0)][:, cam, ...]
                    # if scale > 0:
                    #     rgb_gt = F.interpolate(rgb_gt, scale_factor=1.0/(2**scale), mode='bilinear', align_corners=False)
                    image = outputs[('cam', cam)][('gaussian_color', frame_id, scale)]
                    # destination folder: test_images/<token>/
                    dest_base = test_images_dir / token
                    self.save_image(image, dest_base / f"{cam}.png")
                    self.save_image(rgb_gt, dest_base / f"{cam}_gt.png")
                    if self.novel_view_mode == 'SF':
                        self.save_image(inputs[('color', 0, 0)][:, cam, ...], dest_base / f"{cam}_0_gt.png")
                    elif self.novel_view_mode == 'MF':
                        self.save_image(inputs[('color', -1, 0)][:, cam, ...], dest_base / f"{cam}_prev_gt.png")
                        self.save_image(inputs[('color', 1, 0)][:, cam, ...], dest_base / f"{cam}_next_gt.png")

    def save_image(
        self,
        image: FloatImage,
        path: Union[Path, str],
    ) -> None:
        """Save an image. Assumed to be in range 0-1."""

        # Create the parent directory if it doesn't already exist.
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Save the image.
        Image.fromarray(self.prep_image(image)).save(path)


    def prep_image(self, image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
        # Handle batched images.
        if image.ndim == 4:
            image = rearrange(image, "b c h w -> c h (b w)")

        # Handle single-channel images.
        if image.ndim == 2:
            image = rearrange(image, "h w -> () h w")

        # Ensure that there are 3 or 4 channels.
        channel, _, _ = image.shape
        if channel == 1:
            image = repeat(image, "() h w -> c h w", c=3)
        assert image.shape[0] in (3, 4)

        image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
        return rearrange(image, "c h w -> h w c").cpu().numpy()

    @torch.no_grad()
    def compute_reconstruction_metrics(self, inputs, outputs, scale=0):
        """
        This function computes reconstruction metrics.
        """
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0
        gaussian_num = 0.0
        render_time = 0.0 
        if self.novel_view_mode == 'SF':
            frame_id = 1
        elif self.novel_view_mode == 'MF':
            frame_id = 0
        else:
            raise ValueError(f"Invalid novel view mode: {self.novel_view_mode}")
        
        next_extr = []
        for cam in range(self.num_cams):
            rgb_gt = inputs[('color', frame_id, 0)][:, cam, ...]
            # if scale > 0:
            #     rgb_gt = F.interpolate(rgb_gt, scale_factor=1.0/(2**scale), mode='bilinear', align_corners=False)
            image = outputs[('cam', cam)][('gaussian_color', frame_id, scale)]
            gaussian_num += outputs[('cam', cam)][('xyz', 0, 3)].shape[1]
            render_time += outputs[('cam', cam)][('render_time', -1, scale)]
            psnr += self.compute_psnr(rgb_gt, image).mean()
            ssim += self.compute_ssim(rgb_gt, image).mean()
            lpips += self.compute_lpips(rgb_gt, image).mean()

            mask_img = outputs[('cam', cam)][('mask_color', 0, 0)]
            # Downsample image by 0.8x before computing metrics
            image_downsampled = F.interpolate(image, scale_factor=1.25, mode='bilinear', align_corners=False)
            image_downsampled = F.interpolate(image_downsampled, scale_factor=0.8, mode='bilinear', align_corners=False)
            if self.save_images:
                assert self.eval_batch_size == 1
                if self.novel_view_mode == 'SF':
                    self.save_image(mask_img, Path(self.save_path) / inputs['token'][0] / f"{cam}_mask.png")
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', 0, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_0_gt.png")
                    self.save_image(image_downsampled, Path(self.save_path) / inputs['token'][0] / f"{cam}_downsampled.png")
                elif self.novel_view_mode == 'MF':
                    self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                    self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                    self.save_image(inputs[('color', -1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_prev_gt.png")
                    self.save_image(inputs[('color', 1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_next_gt.png")
            next_extr.append(outputs[('cam', cam)][('camf2world', 1, 0)][0])
        extrinsics = inputs.get('extrinsics', None)
        if extrinsics is not None:
            try:
                if hasattr(extrinsics, 'detach'):
                    ex_list = extrinsics.detach().cpu().numpy().tolist()
                else:
                    ex_list = np.array(extrinsics).tolist()
                next_extr_list = [extr.cpu().numpy().tolist() for extr in next_extr]
                dest = Path(self.save_path) / inputs['token'][0]
                dest.mkdir(parents=True, exist_ok=True)
                # Save both original extrinsics and computed next_extr into a single JSON
                out_dict = {
                    'extrinsics': ex_list,
                    'next_extr': next_extr_list
                }
                with open(dest / 'extrinsics.json', 'w') as f:
                    json.dump(out_dict, f)
            except Exception as e:
                print(f"Failed to save extrinsics for {inputs['token'][0]}: {e}")
        
        psnr /= self.num_cams
        ssim /= self.num_cams
        lpips /= self.num_cams

        
        metrics_json_path = Path(self.save_path) / inputs['token'][0]/"metrics.json"

        with open(metrics_json_path, 'w') as f:
            json.dump(
                {
                    'psnr': float(psnr),
                    'ssim': float(ssim),
                    'lpips': float(lpips),
                },
                f,
                indent=2,
            )
        print(f"Saved PSNR/SSIM/LPIPS metrics to {metrics_json_path}")

        return psnr, ssim, lpips, gaussian_num, render_time
    
    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ground_truth = ground_truth.clip(min=0, max=1)
        predicted = predicted.clip(min=0, max=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * mse.log10()
    
    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        value = self.lpips.forward(ground_truth, predicted, normalize=True)
        return value[:, 0, 0, 0]
    
    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ssim = [
            structural_similarity(
                gt.detach().cpu().numpy(),
                hat.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
