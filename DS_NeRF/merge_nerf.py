import os

import numpy as np
import imageio
import torch
from tqdm import tqdm
from run_nerf_helpers_tcnn import NeRF_TCNN
from run_nerf_helpers import *
from correspondence_utils import *

from load_llff import load_llff_data
from camera_pose_visualizer import CameraPoseVisualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
np.random.seed(0)

def visualize(poses, points, save_path):
    visualizer = CameraPoseVisualizer([-1, 1], [-1, 1], [-2, 0])
    pyramid_color = (1, 0, 0, 0.3)
    points_color  = (0, 0, 0, 0.5)
    line_color    = (0, 0, 1, 1.0)

    poses_for_plot = poses.copy()
    poses_for_plot[:, :, 2] *= -1

    visualizer.ax.view_init(120, -100)

    # Plot Pose Pyramids
    visualizer.plot_poses(poses_for_plot, color=pyramid_color, focal_len_scaled=0.2)
    
    # Plot Object Points
    visualizer.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color=points_color)
    
    # Plot a Line from the Object Mean to the Bottom 
    mean = points.mean(0)
    bottom = np.percentile(points[:, 1], 5)
    visualizer.ax.plot([mean[0], mean[0]], [mean[1], bottom], [mean[2], mean[2]], color=line_color, lw=5, marker='x', ms=10, mew=3)

    visualizer.fig.savefig(save_path)
    return

rot_phi = lambda phi : np.array([
    [np.cos(phi),0,np.sin(phi)],
    [0,1,0],
    [-np.sin(phi),0,np.cos(phi)],
], dtype=np.float32)

rot_theta = lambda theta : np.array([
    [1,0,0],
    [0,np.cos(theta),np.sin(theta)],
    [0,-np.sin(theta),np.cos(theta)],
], dtype=np.float32)

def apply_transform(pts, transform):
    dtype=pts.dtype
    
    if "origin" in transform:
        origin = transform["origin"]
    else:
        origin = [0, 0, 0]
    origin = torch.Tensor(origin).to(dtype=dtype, device=device)[None, None]

    if "scale" in transform:
        scale = transform["scale"]
    else:
        scale = 1.

    if "rotat" in transform:
        rotat = transform["rotat"]
    else:
        rotat = [0, 0]
    phi, theta = rotat
    rotat = rot_phi(-phi/180.*np.pi) @ rot_theta(-theta/180.*np.pi)
    rotat = torch.Tensor(rotat).to(dtype=dtype, device=device)

    if "trans" in transform:
        trans = transform["trans"]
    else:
        trans = [0, 0, 0]
    trans = torch.Tensor(trans).to(dtype=dtype, device=device)[None, None]

    return origin + torch.sum((pts - origin)[..., None, :] * rotat, -1) / scale - trans

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024 * 32, c2w=None, near=0., far=1., **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
    """
    rays_o, rays_d = get_rays(H, W, focal, c2w)

    # provide ray directions as input
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)  # B x 11

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    return ret_list


def render_path(render_poses, hwf, chunk, render_kwargs, render_factor=0):
    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    K = np.array([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1]
    ])

    rgbs = []
    accs = []
    disps = []

    for c2w in tqdm(render_poses):
        with torch.no_grad():
            rgb, disp, acc = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)

        rgbs.append(rgb.detach().cpu().numpy())
        accs.append(acc.detach().cpu().numpy())
        disps.append(disp.detach().cpu().numpy())

    rgbs = np.stack(rgbs, 0)
    accs = np.stack(accs, 0)
    disps = np.stack(disps, 0)

    rgbs = rgbs * accs[..., None]

    return rgbs, disps

def create_nerf_tcnn(args):
    """Instantiate NeRF's MLP model.
    """
    id_fn = lambda x: x
    
    object_model = NeRF_TCNN(encoding="hashgrid")
    object_model_fine = NeRF_TCNN(encoding="hashgrid")

    scene_model = NeRF_TCNN(encoding="hashgrid")
    scene_model_fine = NeRF_TCNN(encoding="hashgrid")

    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(inputs, viewdirs, network_fn, embed_fn=id_fn, embeddirs_fn=id_fn, netchunk=args.netchunk)

    basedir = args.basedir
    object_expname = args.object_expname
    scene_expname = args.scene_expname
    ##########################

    # Load checkpoints
    object_ckpts = [os.path.join(basedir, object_expname, f) for f in sorted(os.listdir(os.path.join(basedir, object_expname))) if 'tar' in f]
    scene_ckpts = [os.path.join(basedir, scene_expname, f) for f in sorted(os.listdir(os.path.join(basedir, scene_expname))) if 'tar' in f]

    assert len(object_ckpts) > 0 and len(scene_ckpts) > 0

    object_ckpt_path = object_ckpts[-1]
    object_ckpt = torch.load(object_ckpt_path)

    scene_ckpt_path = scene_ckpts[-1]
    scene_ckpt = torch.load(scene_ckpt_path)

    # Load model
    object_model.load_state_dict(object_ckpt['network_fn_state_dict'])
    object_model_fine.load_state_dict(object_ckpt['network_fine_state_dict'])

    scene_model.load_state_dict(scene_ckpt['network_fn_state_dict'])
    scene_model_fine.load_state_dict(scene_ckpt['network_fine_state_dict'])
    ##########################

    render_kwargs = {
        'N_samples': args.N_samples,
        'N_importance': args.N_importance,
        'network_query_fn': network_query_fn,
        'object_network_fn': object_model,
        'object_network_fine': object_model_fine,
        'scene_network_fn': scene_model,
        'scene_network_fine': scene_model_fine,
        'white_bkgd': args.white_bkgd,
        'lindisp': args.lindisp
    }

    return render_kwargs

def render_rays(ray_batch,
                network_query_fn,
                N_samples,
                object_network_fn,
                scene_network_fn,
                N_importance=0,
                object_network_fine=None,
                scene_network_fine=None,
                lindisp=False,
                white_bkgd=False,
                transform=None,
                ):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      object_network_fn: function. Model for predicting RGB and density of object at each point
        in space.
      scene_network_fn: function. Model for predicting RGB and density of scene at each point
        in space.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      object_network_fine: "fine" network with same spec as object_network_fn.
      scene_network_fine: "fine" network with same spec as scene_network_fn.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      white_bkgd: bool. If True, assume a white background.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, 8:11]
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    if transform is not None:
        pts = apply_transform(pts, transform)
    object_raw = network_query_fn(pts, viewdirs, object_network_fn)
    rgb_map, disp_map, acc_map, object_weights, _, _ = raw2outputs(object_raw, z_vals, rays_d, white_bkgd=white_bkgd)

    scene_raw = network_query_fn(pts, viewdirs, scene_network_fn)
    rgb_map, disp_map, acc_map, scene_weights, _, _ = raw2outputs(scene_raw, z_vals, rays_d, white_bkgd=white_bkgd)

    if N_importance > 0:
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])

        object_z_samples = sample_pdf(z_vals_mid, object_weights[..., 1:-1], N_importance, det=True)
        object_z_samples = object_z_samples.detach()
        object_z_vals, _ = torch.sort(torch.cat([z_vals, object_z_samples], -1), -1)
        object_pts = rays_o[..., None, :] + rays_d[..., None, :] * object_z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        if transform is not None:
            object_pts = apply_transform(object_pts, transform)
        object_raw = network_query_fn(object_pts, viewdirs, object_network_fine)

        scene_z_samples = sample_pdf(z_vals_mid, scene_weights[..., 1:-1], N_importance, det=True)
        scene_z_samples = scene_z_samples.detach()
        scene_z_vals, _ = torch.sort(torch.cat([z_vals, scene_z_samples], -1), -1)
        scene_pts = rays_o[..., None, :] + rays_d[..., None, :] * scene_z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        scene_raw = network_query_fn(scene_pts, viewdirs, scene_network_fine)

        merged_z_vals, z_order = torch.sort(torch.cat([object_z_vals, scene_z_vals], -1), -1)
        
        merged_raw = torch.cat([object_raw, scene_raw], 1)

        _b, _n, _c = merged_raw.shape

        merged_raw = merged_raw[
            torch.arange(_b).view(_b, 1, 1).repeat(1, _n, _c),
            z_order.view(_b, _n, 1).repeat(1, 1, _c),
            torch.arange(_c).view(1, 1, _c).repeat(_b, _n, 1),
        ]

        # merged_pts = torch.cat([object_pts, scene_pts], 1)
        # merged_pts = merged_pts[
        #     torch.arange(_b).view(_b, 1, 1).repeat(1, _n, 3),
        #     z_order.view(_b, _n, 1).repeat(1, 1, 3),
        #     torch.arange(3).view(1, 1, 3).repeat(_b, _n, 1),
        # ]
        # inside_pts = ((merged_pts - origin)**2).sum(-1) < 0.01
        # merged_raw[inside_pts] = torch.Tensor([1, 0, 0, 1000]).to(dtype=merged_raw.dtype, device=merged_raw.device)

        rgb_map, disp_map, acc_map, _, _, _ = raw2outputs(merged_raw, merged_z_vals, rays_d, white_bkgd=white_bkgd)
    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--object_expname", type=str,
                        help='object experiment name')
    parser.add_argument("--scene_expname", type=str,
                        help='scene experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=0.01,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=float, default=10,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_test_ray", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true',
                        help='render the train set instead of render_poses path')
    parser.add_argument("--render_mypath", action='store_true',
                        help='render the test path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=1000000,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=100000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')
    parser.add_argument("--colmap_depth", action='store_true',
                        help="Use depth supervision by colmap.")
    parser.add_argument("--depth_loss", action='store_true',
                        help="Use depth supervision by colmap - depth loss.")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help="Depth lambda used for loss.")
    parser.add_argument("--sigma_loss", action='store_true',
                        help="Use depth supervision by colmap - sigma loss.")
    parser.add_argument("--sigma_lambda", type=float, default=0.1,
                        help="Sigma lambda used for loss.")
    parser.add_argument("--weighted_loss", action='store_true',
                        help="Use weighted loss by reprojection error.")
    parser.add_argument("--relative_loss", action='store_true',
                        help="Use relative loss.")
    parser.add_argument("--depth_with_rgb", action='store_true',
                        help="single forward for both depth and rgb")
    parser.add_argument("--normalize_depth", action='store_true',
                        help="normalize depth before calculating loss")

    parser.add_argument("--no_tcnn", action='store_true',
                        help='set to not use tinycudann and use the original NeRF')

    parser.add_argument("--clf_weight", type=float, default=0.01,
                        help='The weight of the classification loss')
    parser.add_argument("--clf_reg_weight", type=float, default=0.01,
                        help='The weight of the classification regularizer')
    parser.add_argument("--feat_weight", type=float,
                        default=0.01, help='The weight of the feature loss')
    parser.add_argument("--i_feat", type=int, default=10,
                        help='frequency of calculating the feature loss')
    parser.add_argument("--prepare", action='store_true',
                        help='Prepare depths for inpainting')
    parser.add_argument("--lpips", action='store_true',
                        help='use perceptual loss for rgb inpainting')
    parser.add_argument("--N_gt", type=int, default=0,
                        help='Number of ground truth inpainted samples')
    parser.add_argument("--N_train", type=int, default=None,
                        help='Number of training images used for optimization')
    parser.add_argument("--train_gt", action='store_true',
                        help='Use the gt inpainted images to train a NeRF')
    parser.add_argument("--masked_NeRF", action='store_true',
                        help='Only train NeRF on unmasked pixels')
    parser.add_argument("--object_removal", action='store_true',
                        help='Remove the object and shrink the masks')
    parser.add_argument("--segmented_NeRF", action='store_true',
                        help='Train NeRF for the masked region')
    parser.add_argument("--tmp_images", action='store_true',
                        help='Use images in lama_images_tmp for ablation studies')
    parser.add_argument("--no_geometry", action='store_true',
                        help='Stop using inpainted depths for training')

    parser.add_argument("--lpips_render_factor", type=int, default=2,
                        help='The stride (render factor) used for sampling patches for the perceptual loss')
    parser.add_argument("--patch_len_factor", type=int, default=8,
                        help='The resizing factor to obtain the side lengths of the patches for the perceptual loss')
    parser.add_argument("--lpips_batch_size", type=int, default=4,
                        help='The number of patches used in each iteration for the perceptual loss')

    parser.add_argument("--in2n", action='store_true')
    parser.add_argument("--base_expname", type=str, default="None",
                        help='base experiment name. only used for in2n')
    parser.add_argument("--prompt", type=str, default="Don't change the image")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--lower_bound", type=float, default=0.02)
    parser.add_argument("--upper_bound", type=float, default=0.98)
    parser.add_argument("--lpips_lambda", type=float, default=0.01)

    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--apply_all_transforms", action="store_true")
    parser.add_argument("--render_staticcam", action="store_true")
    return parser

def main():
    parser = config_parser()
    args = parser.parse_args()

    _, poses, bds, render_poses, _, _, _, _, object_points = load_llff_data(args.datadir,
                                                             args.factor,
                                                             recenter=True,
                                                             bd_factor=.75,
                                                             spherify=args.spherify,
                                                             return_object_points=True,
                                                             args=args)
    
    hwf = poses[0, :3, -1]
    print('Loaded llff', render_poses.shape, hwf)

    print('DEFINING BOUNDS')
    near = np.ndarray.min(bds) * .9
    far = np.ndarray.max(bds) * 1.
    print('NEAR FAR', near, far)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir
    basedir = args.basedir
    expname = args.expname

    logdir = os.path.join(basedir, expname)
    os.makedirs(logdir, exist_ok=True)

    # Create nerf model
    render_kwargs = create_nerf_tcnn(args)

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs.update(bds_dict)

    visualize(poses[:, :3, :4], object_points, os.path.join(logdir, "object_points.png"))
    object_mean = object_points.mean(0)
    print("object_mean", object_mean)
    bottom = np.percentile(object_points[:, 1], 5)
    linewidth = np.abs(object_mean[1] - bottom)

    render_kwargs["transform"] = {"origin": object_mean}
    render_kwargs["transform"]["scale"] = args.scale
    render_kwargs["transform"]["trans"] = [0, -(1-args.scale)*linewidth, 0]

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    with torch.no_grad():
        print('render poses shape', render_poses.shape)
        if args.apply_all_transforms:
            rgbs_all = []
            disps_all = []
            for i in tqdm(range(720)):
                render_pose = render_poses[i:i+1]
                render_kwargs["transform"] = {"origin": object_mean}
                if i < 240:
                    render_kwargs["transform"]["scale"] = 1 - 0.2 * np.sin(i/120*np.pi)
                elif i < 360:
                    render_kwargs["transform"]["rotat"] = [-15 * np.sin(i/60*np.pi), 0]
                elif i < 480:
                    render_kwargs["transform"]["rotat"] = [0, -15 * np.sin(i/60*np.pi)]
                else:
                    render_kwargs["transform"]["trans"] = [0.5 * np.sin(i/120*np.pi), 0.5 * (1-np.cos(i/120*np.pi)), 0]
                rgbs, disps = render_path(render_pose, hwf, args.chunk, render_kwargs, render_factor=args.render_factor)
                rgbs_all.append(rgbs)
                disps_all.append(disps)
            rgbs = np.concatenate(rgbs_all, 0)
            disps = np.concatenate(rgbs_all, 0)
        else:
            rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs, render_factor=args.render_factor)
        print('Done rendering')
        imageio.mimwrite(os.path.join(
            logdir, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
        disps[np.isnan(disps)] = 0
        print('Depth stats', np.mean(disps), np.max(
            disps), np.percentile(disps, 95))
        imageio.mimwrite(os.path.join(logdir, 'disp.mp4'), to8b(disps / np.percentile(disps, 95)), fps=30, quality=8)

        return

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    main()
