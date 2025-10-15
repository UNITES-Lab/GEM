import itertools
import math
import os
from code.dataset.dataset import EMPIARDataset
from code.model.deformation import ImageEncoder  # Added import

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import rich
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import umap
from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn)
from scipy.stats import special_ortho_group
from sklearn.neighbors import NearestNeighbors
from spdl.dataloader import get_pytorch_dataloader
from torch.utils.data import DataLoader
from torch_fourier_shift import fourier_shift_image_2d
from torchmetrics.functional.image import total_variation as tv
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings, GaussianRasterizer,
    GaussianVoxelizationSettings, GaussianVoxelizer)

from main import Args as MainArgs
from simple_gaussian import GaussianModel, OptimizationParams
from simple_gaussian_utils import ssim

# torch.autograd.set_detect_anomaly(True)

def knn(x: torch.Tensor, K: int = 4) -> torch.Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)

def inverse_sigmoid(x):
    return torch.log(x / (1 - x + 1e-6))

def density_inverse_activation(x, beta=1):
    return torch.log(torch.exp(beta * x) - 1) / beta

scaling_inverse_activation = lambda x: inverse_sigmoid(torch.relu((x - 0.0005) / (0.5 - 0.0005)))

scaling_activation = lambda x: torch.sigmoid(x) * (0.5 - 0.005) + 0.005

# Define DeformationMLP
class DeformationMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_hidden_layers=3, output_dim_xyz=3, output_dim_density=1, output_dim_scale=3, output_dim_rotation=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        
        self.base_mlp = nn.Sequential(*layers)
        
        self.fc_xyz = nn.Linear(hidden_dim, 1)
        self.fc_density = nn.Linear(hidden_dim, output_dim_density)
        self.fc_scale = nn.Linear(hidden_dim, output_dim_scale)
        self.fc_rotation = nn.Linear(hidden_dim, output_dim_rotation)

    def forward(self, x):
        x_base = self.base_mlp(x)
        dx = torch.tanh(self.fc_xyz(x_base))            # (B, 1)
        zeros = torch.zeros_like(dx).expand(-1, 2)      # (B, 2) → [0, 0]
        delta_xyz = torch.cat([dx, zeros], dim=-1)      # (B, 3) → [dx, 0, 0]
        delta_density = self.fc_density(x_base)
        delta_scale = self.fc_scale(x_base)
        delta_rotation = self.fc_rotation(x_base)
        
        return delta_xyz, delta_density, delta_scale, delta_rotation


def main(
    args: MainArgs,
    lr: float = 1e-3,
    mlp_lr: float = 1e-3, # Added learning rate for MLP
):
    # Set random seed and prepare save directory
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    opt = OptimizationParams()

    # Initialize dataset and dataloader
    ds = EMPIARDataset(
        mrcs=os.path.join(args.dataset_dir, "particles.mrcs") if args.particles is None else args.particles,
        ctf=os.path.join(args.dataset_dir, "ctf.pkl") if args.ctf is None else args.ctf,
        poses=os.path.join(args.dataset_dir, "poses.pkl") if args.poses is None else args.poses,
        args=args,
        size=args.size,
        sign=args.sign,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=16) # shuffle was False, kept it
    opt.iterations = len(ds) * args.epochs
        
    # Initialize model and optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaussianModel(scale_bound=np.array([0.001, 1]), num_points=50000) # num_points is initial, can change
    model.training_setup(opt)
    
    # # Initialize ImageEncoder and DeformationMLP
    # image_encoder = ImageEncoder(encoder_type=args.hetero_encoder_type, latent_dim=args.hetero_latent_dim,
    #                              size=args.size, hartley=args.hartley).to(device)
    # deformation_mlp = DeformationMLP(input_dim=3 + args.hetero_latent_dim, hidden_dim=128, num_hidden_layers=3).to(device)
    
    # optimizer_deformation = optim.Adam(list(image_encoder.parameters()) + list(deformation_mlp.parameters()), lr=mlp_lr)

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        model.restore(ckpt['model_args'], opt)
        global_step = ckpt.get('global_step', 0)
        # if 'image_encoder_state_dict' in ckpt:
        #     image_encoder.load_state_dict(ckpt['image_encoder_state_dict'])
        # if 'deformation_mlp_state_dict' in ckpt:
        #     deformation_mlp.load_state_dict(ckpt['deformation_mlp_state_dict'])
        # if 'optimizer_deformation_state_dict' in ckpt:
        #     optimizer_deformation.load_state_dict(ckpt['optimizer_deformation_state_dict'])
        print(f"Loaded checkpoint {args.load_ckpt}, starting from step {global_step}")
    else:
        global_step = 0
        
    if args.load_embd:
        all_latent_features_np = np.load(args.load_embd)
    
    losses = []
    # Training loop with samples/sec metric
    if not args.val_only:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),     
            TimeElapsedColumn(),
            TaskProgressColumn(show_speed=True)
        )
        with progress:
            task_description = f"Training Epoch {1}/{args.epochs}" if args.epochs > 0 else "Training"
            task = progress.add_task(task_description, total=None)

            for epoch in range(args.epochs):
                if args.epochs > 0:
                    progress.update(task, description=f"Training Epoch {epoch+1}/{args.epochs}")

                for i, batch in enumerate(loader):
                    # Move inputs to device
                    R_batch = batch["rotations"].to(device)
                    t_batch = batch["translations"].to(device)
                    imgs_batch = batch["images"].to(device) # Shape (B, H, W)
                    enc_imgs = batch["enc_images"].to(device)
                    ctfs_batch = batch["ctfs"].to(device)
                    
                    model.update_learning_rate(global_step) # Pass global_step for LR schedulers

                    # Get current Gaussian parameters (these are shared before deformation)
                    # Detach them if their gradients should not flow through MLP back to themselves directly,
                    # but rather through the addition of deltas.
                    # Gradients will flow to original params via the addition: final = original + delta.
                    _original_xyz = model.get_xyz 
                    _original_density = model.get_density
                    _original_scales = model.get_scaling
                    _original_rotations = model.get_rotation
                    
                    num_points = _original_xyz.shape[0]
                    
                    final_xyz = _original_xyz
                    final_density = _original_density
                    final_scales = _original_scales
                    final_rotations = _original_rotations
                        
                    R = R_batch[0]  # Shape from (1,...) to (...)
                    t = t_batch[0]
                    img_gt = imgs_batch[0] # Shape (H, W), from (1,H,W)
                    ctf = ctfs_batch[0]   # Shape (H, W), from (1,H,W)
                    
                    viewmat = torch.eye(4, device=R.device)
                    viewmat[:3, :3] = R.mT
                    viewmat[3, 2] = 5
                    
                    screenspace_points = torch.zeros_like(final_xyz, dtype=final_xyz.dtype, device=final_xyz.device, requires_grad=True)
                    screenspace_points.retain_grad()
            
                    raster_settings = GaussianRasterizationSettings(
                        image_height=int(args.size), image_width=int(args.size),
                        tanfovx=1., tanfovy=1., scale_modifier=1.,
                        viewmatrix=viewmat, projmatrix=viewmat, 
                        campos=torch.tensor([0, 0, 1], dtype=torch.float32, device=device),
                        prefiltered=False, mode=0, debug=False,
                    )
                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                    
                    # pred_image is expected to be (1, H, W)
                    pred_image, radii = rasterizer( 
                        means3D=final_xyz, means2D=screenspace_points,
                        opacities=final_density, scales=final_scales,
                        rotations=final_rotations, cov3D_precomp=None,
                    )
                    visibility_filter = radii > 0
                    
                    # Apply CTF corruption
                    # pred_image is (1,H,W), ctf is (H,W). Unsqueeze ctf for broadcasting.
                    corrupted_pred_image = torch.fft.fftshift(
                        torch.fft.irfft2(
                            torch.fft.rfft2(torch.fft.ifftshift(pred_image)) * torch.fft.fftshift(ctf.unsqueeze(0))[..., :args.size // 2 + 1]
                        )
                    )
                    
                    # Ground truth image processing
                    # img_gt is (H,W). Unsqueeze to (1,H,W) for fourier_shift_image_2d and loss.
                    gt_img_shifted = fourier_shift_image_2d(img_gt.unsqueeze(0), torch.flip(t[:2], dims=[-1]) * args.size)

                    # Assign to tensors used in loss calculation (these names were used post-concatenation previously)
                    pred_images_tensor = pred_image
                    corrupted_pred_images_tensor = corrupted_pred_image
                    gt_images_shifted_tensor = gt_img_shifted
                    
                    loss_recon = F.mse_loss(corrupted_pred_images_tensor, gt_images_shifted_tensor)
                    
                    # Original scale regularization based on undeformed scales from the model
                    # This encourages the base Gaussians to have reasonable scales.
                    # Deformation MLP can then refine them.
                    current_model_scales = model.get_scaling 
                    if current_model_scales.shape[0] > 0:
                        scale_norms = torch.norm(current_model_scales, dim=1)
                        min_scale_norm = scale_norms.min().item()
                        max_scale_norm = scale_norms.max().item()
                        loss_scale_reg = 0.1 * max_scale_norm
                    else:
                        min_scale_norm = 0.0
                        max_scale_norm = 0.0
                        loss_scale_reg = 0.0
                    
                    loss = loss_recon

                    loss.backward()
                    
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)

                    # optimizer_deformation.step()
                    # optimizer_deformation.zero_grad(set_to_none=True)

                    if global_step % 5000 == 0: # Original logging frequency for loss values
                        losses.append(loss.item())
                    
                    # with torch.no_grad():
                    #     # Adaptive control (original code was commented out, keeping it so)
                    #     # If enabled, ensure it uses appropriate values (e.g., from the first item in batch)
                    #     model.max_radii2D[visibility_filter] = torch.max(
                    #         model.max_radii2D[visibility_filter], radii[visibility_filter]
                    #     )
                    #     model.add_densification_stats(screenspace_points, visibility_filter)
                    #     if global_step < 10000 and global_step > 500 and global_step % 100 == 0: # Changed to global_step
                    #         model.densify_and_prune(
                    #             opt.densify_grad_threshold, 
                    #             opt.density_min_threshold,
                    #             opt.max_screen_size,
                    #             opt.max_scale,
                    #             opt.max_num_gaussians,
                    #             opt.densify_scale_threshold,
                    #             torch.tensor([[-1., -1., -1.], [1., 1., 1.]], device=device),
                    #         )
                    #     if model.get_density.shape[0] == 0 and not args.val_only : # Check num_points after pruning
                    #         print("Warning: No Gaussians left. Consider adjusting densification/pruning or initialization.")

                    progress.update(task, advance=1) # Advance per batch

                    if global_step % args.log_vis_step == 0:
                        rich.print(f"Epoch {epoch+1}, Iter {global_step:06d}, Loss: {loss.item():.4f}, Scale Norm Max: {max_scale_norm:.4f}, Num Gaussians: {num_points}")
                        # Visualize using the first image of the batch
                        out_vis = pred_images_tensor[0].detach().cpu().numpy()
                        gt_vis = imgs_batch[0].detach().cpu().numpy()
                        cr_out_vis = corrupted_pred_images_tensor[0].detach().cpu().numpy()
                        
                        plt.imsave(os.path.join(args.save_dir, f"{global_step:06d}_pr.png"), out_vis, cmap="gray")
                        plt.imsave(os.path.join(args.save_dir, f"{global_step:06d}_cr.png"), cr_out_vis, cmap="gray")
                        plt.imsave(os.path.join(args.save_dir, f"{global_step:06d}_gt.png"), gt_vis, cmap="gray")

                    global_step += 1
                    if args.ckpt_save_step > 0 and global_step % args.ckpt_save_step == 0:
                        os.makedirs(args.save_dir, exist_ok=True)
                        ckpt_data = {
                            'model_args': model.capture(),
                            'training_args': opt, # opt are params for GaussianModel training setup
                            'global_step': global_step,
                            # 'image_encoder_state_dict': image_encoder.state_dict(),
                            # 'deformation_mlp_state_dict': deformation_mlp.state_dict(),
                            # 'optimizer_deformation_state_dict': optimizer_deformation.state_dict(),
                            'main_args': vars(args), # Save main arguments
                            'lr': lr,
                            'mlp_lr': mlp_lr,
                            'latent_dim': args.hetero_latent_dim
                        }
                        ckpt_path = os.path.join(args.save_dir, f"{global_step:06d}.pth")
                        torch.save(ckpt_data, ckpt_path)
                        print(f"Saved checkpoint: {ckpt_path}")
   
    # # First, collect all latent features if in hetero mode
    # all_latent_features_np = None
    # if args.hetero and ds is not None and image_encoder is not None:
    #     print("Generating latent features for UMAP and Voxelization...")
    #     if args.load_embd:
    #         all_latent_features = np.load(args.load_embd)
    #     else:
    #         all_latent_features = []
            
    #         # Ensure dataset is available
    #         if 'ds' not in locals() and 'ds' not in globals():
    #             ds_for_latent = EMPIARDataset(
    #                 mrcs=os.path.join(args.dataset_dir, "particles.mrcs") if args.particles is None else args.particles,
    #                 ctf=os.path.join(args.dataset_dir, "ctf.pkl") if args.ctf is None else args.ctf,
    #                 poses=os.path.join(args.dataset_dir, "poses.pkl") if args.poses is None else args.poses,
    #                 args=args,
    #                 size=args.size,
    #                 sign=args.sign,
    #             )
    #         else:
    #             ds_for_latent = ds

    #         # It's better to use a DataLoader for efficient loading
    #         latent_loader = DataLoader(ds_for_latent, batch_size=32, shuffle=False, num_workers=4)

    #         image_encoder.eval() # Set encoder to evaluation mode
    #         with torch.no_grad():
    #             for batch in latent_loader:
    #                 imgs_batch = batch["enc_images"].to(device) # Shape (1, H, W)
    #                 # ImageEncoder expects (B, C, H, W)
    #                 latent_features = image_encoder(imgs_batch.unsqueeze(1)) # (1, latent_dim)
    #                 all_latent_features.append(latent_features.cpu().numpy())
        
    #     if not args.load_embd:
    #         all_latent_features_np = np.concatenate(all_latent_features, axis=0) # Shape (N, latent_dim)
            
    #         # Save all latent features
    #         latent_save_path = os.path.join(args.save_dir, "latent_features.npy")
    #         np.save(latent_save_path, all_latent_features_np)
    #         print(f"Saved all latent features to {latent_save_path}")
    #     else:
    #         all_latent_features_np = all_latent_features

    # Voxelization part (after training)
    with torch.no_grad():
        voxel_settings = GaussianVoxelizationSettings(
            scale_modifier=1.,
            nVoxel_x=int(args.size), nVoxel_y=int(args.size), nVoxel_z=int(args.size),
            sVoxel_x=float(2), sVoxel_y=float(2), sVoxel_z=float(2),
            center_x=float(0), center_y=float(0), center_z=float(0),
            prefiltered=False,
            debug=False,
        )
        voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

        # Get the base model's current state
        _original_xyz = model.get_xyz
        _original_density = model.get_density
        _original_scales = model.get_scaling
        _original_rotations = model.get_rotation
        
        if _original_xyz.shape[0] > 0:
            if args.hetero and all_latent_features_np is not None:
                # Hetero case: sample some latent features and generate a volume for each
                num_samples = min(6, len(all_latent_features_np)) # e.g., sample up to 10
                sample_indices = np.random.choice(len(all_latent_features_np), num_samples, replace=False)
                
                print(f"Generating {num_samples} deformed volumes from sampled latent features...")
                for i, idx in enumerate(sample_indices):
                    latent_feature = torch.from_numpy(all_latent_features_np[idx]).to(device)
                    
                    # Deform the Gaussians using the sampled latent feature
                    num_points = _original_xyz.shape[0]
                    expanded_latent_feature = latent_feature.unsqueeze(0).repeat(num_points, 1)
                    mlp_input = torch.cat([_original_xyz, expanded_latent_feature], dim=1)
                    d_xyz, d_density, d_scales, d_rotations = deformation_mlp(mlp_input)

                    final_xyz = _original_xyz + d_xyz
                    # final_density = torch.clamp(_original_density + d_density, 0.0, 1.0)
                    # final_scales = torch.clamp(_original_scales + d_scales, 0.005, 0.5)
                    # final_rotations = F.normalize(_original_rotations + d_rotations, p=2, dim=-1)
                    final_density = _original_density
                    final_scales = _original_scales
                    final_rotations = _original_rotations

                    volume, _ = voxelizer(
                        means3D=final_xyz,
                        opacities=final_density,
                        scales=final_scales,
                        rotations=final_rotations,
                        cov3D_precomp=None,
                    )
                    
                    print(f"  Voxelized volume {i+1}/{num_samples}, shape: {volume.shape}")
                    filename = f"volume_deformed_sample_{i:03d}.mrc"
                    with mrcfile.new(os.path.join(args.save_dir, filename), overwrite=True) as mrc:
                        density_volume = volume.cpu().numpy().astype(np.float32)
                        density_volume = np.rot90(density_volume, k=3, axes=(1, 2))
                        density_volume = np.rot90(density_volume, k=2, axes=(0, 2))
                        density_volume = np.rot90(density_volume, k=3, axes=(0, 1))
                        mrc.set_data(density_volume)
                        mrc.set_volume()
                        mrc.voxel_size = ds.Apix
                        mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = 0,0,0
            else:
                # Original non-hetero case: voxelize the base model
                volume, _ = voxelizer(
                    means3D=_original_xyz,
                    opacities=_original_density,
                    scales=_original_scales,
                    rotations=_original_rotations,
                    cov3D_precomp=None,
                )
                print(f"Voxelized base model volume shape: {volume.shape}")
                with mrcfile.new(os.path.join(args.save_dir, "volume_base_model.mrc"), overwrite=True) as mrc:
                    density_volume = volume.cpu().numpy().astype(np.float32)
                    density_volume = np.rot90(density_volume, k=3, axes=(1, 2))
                    density_volume = np.rot90(density_volume, k=2, axes=(0, 2))
                    density_volume = np.rot90(density_volume, k=3, axes=(0, 1))
                    mrc.set_data(density_volume)
                    mrc.set_volume()
                    mrc.voxel_size = ds.Apix
                    mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = 0,0,0
        else:
            print("No Gaussians to voxelize at the end of training.")

    # # UMAP visualization of latent features
    # if all_latent_features_np is not None:
    #     print(f"Collected {all_latent_features_np.shape[0]} latent features for UMAP.")

    #     reducer = umap.UMAP(n_jobs=32)
    #     embedding = reducer.fit_transform(all_latent_features_np)
    #     latent_save_path = os.path.join(args.save_dir, "latent_2d.npy")
    #     np.save(latent_save_path, embedding)
        
    #     plt.figure(figsize=(10, 8))
    #     plt.scatter(embedding[:, 0], embedding[:, 1], s=5, cmap='Spectral', rasterized=True) # s is point size
    #     plt.gca().set_aspect('equal', 'datalim')
    #     plt.title('UMAP projection of latent features', fontsize=14)
    #     umap_save_path = os.path.join(args.save_dir, "latent_features_umap.png")
    #     plt.savefig(umap_save_path)
    #     print(f"Saved UMAP of latent features to {umap_save_path}")
    #     plt.close() # Close the figure to free memory
    # elif args.hetero:
    #     print("No latent features collected for UMAP.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = tyro.cli(MainArgs)
    main(args)
