from dataclasses import dataclass, field
import torch
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.utils.graphics_utils import focal2fov, fov2focal
from argparse import ArgumentParser, Namespace
import os,sys,random
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
# from gaussiansplatting.scene.gaussian_model import BasicPointCloud
from gaussiansplatting.utils.graphics_utils import BasicPointCloud
import numpy as np
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh
import io  
from PIL import Image  
import open3d as o3d

def load_ply(path,save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)


@threestudio.register("gaussiandreamer-system")
class GaussianDreamer(BaseLift3DSystem):
    def __init__(self, cfg, resumed=False) -> None:
        super().__init__(cfg, resumed)
        self.automatic_optimization = False

    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        sh_degree: int = 0
        load_type: int = 0
        load_path: str = "./load/shapes/stand.obj"



    cfg: Config
    def configure(self) -> None:
        self.radius = self.cfg.radius
        self.sh_degree =self.cfg.sh_degree
        self.load_type =self.cfg.load_type
        self.load_path = self.cfg.load_path
        self.viewpoint_stack = []

        self.parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(self.parser)
        args = self.parser.parse_args([])
        # args.save_iterations.append(args.iterations)

        self.gaussian = GaussianModel(lp.extract(args))
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    
    def save_gif_to_file(self,images, output_file):  
        with io.BytesIO() as writer:  
            images[0].save(  
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0  
            )  
            writer.seek(0)  
            with open(output_file, 'wb') as file:  
                file.write(writer.read())
    
    def shape(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        model.load_state_dict(torch.load('./load/shapE_finetuned_with_330kdata.pth', map_location=device)['model_state_dict'])
        diffusion = diffusion_from_config_shape(load_config('diffusion'))

        batch_size = 1
        guidance_scale = 15.0
        prompt = str(self.cfg.prompt_processor.prompt)
        print('prompt',prompt)

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        render_mode = 'nerf' # you can change this to 'stf'
        size = 256 # this is the size of the renders; higher values take longer to render.

        cameras = create_pan_cameras(size, device)

        self.shapeimages = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)

        pc = decode_latent_mesh(xm, latents[0]).tri_mesh()


        skip = 1
        coords = pc.verts
        rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

        coords = coords[::skip]
        rgb = rgb[::skip]

        self.num_pts = coords.shape[0]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb)
        self.point_cloud = point_cloud

        return coords,rgb,0.4
    
    def add_points(self,coords,rgb):
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
        

        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)

        num_points = 1000000  
        points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))


        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)


        points_inside = []
        color_inside= []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < 0.01:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

                
                

        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords,coords],axis=0)
        all_rgb = np.concatenate([all_rgb,rgb],axis=0)
        return all_coords,all_rgb

    def smpl(self):
        self.num_pts  = 50000
        mesh = o3d.io.read_triangle_mesh(self.load_path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=self.num_pts)
        coords = np.array(point_cloud.points)
        shs = np.random.random((self.num_pts, 3)) / 255.0
        rgb = SH2RGB(shs)
        adjusment = np.zeros_like(coords)
        adjusment[:,0] = coords[:,2]
        adjusment[:,1] = coords[:,0]
        adjusment[:,2] = coords[:,1]
        current_center = np.mean(adjusment, axis=0)
        center_offset = -current_center
        adjusment += center_offset
        return adjusment,rgb,0.5
    
    def pcb(self):
        # Since this data set has no colmap data, we start with random points
        if self.load_type==0:
            coords,rgb,scale = self.shape()
        elif self.load_type==1:
            coords,rgb,scale = self.smpl()
        else:
            raise NotImplementedError
        
        bound= self.radius*scale

        all_coords,all_rgb = self.add_points(coords,rgb)

        pcd = BasicPointCloud(points=all_coords *bound, colors=all_rgb, normals=np.zeros((all_coords.shape[0], 3)))

        return pcd
    
    
    def forward(self, batch: Dict[str, Any],renderbackground = None) -> Dict[str, Any]: # 这里可能要改，加入itr/rvq_iter?(看render)

        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        self.viewspace_point_list = []
        for id in range(batch['c2w_3dgs'].shape[0]):
       
            # viewpoint_cam  = Camera(c2w = batch['c2w_3dgs'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])

            c2w = batch['c2w_3dgs'][id]
            # R = c2w[:3, :3]
            R = c2w[:3, :3].float().cpu().numpy()
            T = c2w[:3, 3].float().cpu().numpy()
            

            # 创建空的图像和alpha遮罩
            dummy_image = torch.ones((3, batch['height'], batch['width']), device='cuda')
            dummy_mask = None
            
            viewpoint_cam = Camera(
                colmap_id=-1,  # 使用默认ID
                R=R,
                T=T,
                FoVx=focal2fov(fov2focal(batch['fovy'][id], batch['height']), batch['width']),  # 计算FoVx
                FoVy=batch['fovy'][id],
                image=dummy_image,
                gt_alpha_mask=dummy_mask,
                image_name=f"view_{id}",
                uid=id,
                trans=np.array([0.0, 0.0, 0.0]),
                scale=1.0,
                data_device="cuda"
            )
            if (self.true_global_step - 1) == -1:
                self.pipe.debug = True
            if self.true_global_step <= self.opt.rvq_iter:
                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground, itr=self.true_global_step, rvq_iter=False)
            else:
                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground, itr=self.true_global_step, rvq_iter=True)
            image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            print(f"Rendered image range: min={image.min()}, max={image.max()}")
            
            if id == 0:

                self.radii = radii
            else:


                self.radii = torch.max(radii,self.radii)
                
            
            depth = render_pkg["depth_3dgs"]
            depth =  depth.permute(1, 2, 0)
            
            image =  image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)
            



        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        self.visibility_filter = render_pkg["visibility_filter"]
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def training_step(self, batch, batch_idx):

        print(f"training_step: {self.true_global_step}")
        # 更新高斯模型的学习率
        self.gaussian.update_learning_rate(self.true_global_step)
        
        # 在训练500步后调整引导模型的步数范围
        if self.true_global_step > 500:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)
        
        self.gaussian.update_learning_rate(self.true_global_step)

        # 前向传播，获取渲染结果
        out = self(batch) 
        
        # 获取处理后的提示词
        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]
        
        # 每200步进行一次引导评估
        guidance_eval = (self.true_global_step % 200 == 0)
        
        # 使用引导模型计算损失
        guidance_out = self.guidance(
            images, prompt_utils, **batch, 
            rgb_as_latents=False,
            guidance_eval=guidance_eval
        )

        # 计算总损失
        loss = 0.0
        
        # 1. 添加SDS（Score Distillation Sampling）损失
        loss = loss + guidance_out['loss_sds'] * self.C(self.cfg.loss['lambda_sds'])
        
        # 2. 计算稀疏性损失
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
        
        # 3. 计算不透明度损失
        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        
        print(f"loss: {loss}")

        print("ready for guidance_eval") 

        # 如果是评估步骤，保存评估结果
        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )
        
        print("guidance_out")

        # 记录所有损失参数
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.manual_backward(loss)  #  automatic_optimization = False
        print("backward complete")

        with torch.no_grad():

            if self.true_global_step < 900: # self.opt.densify_until_iter:
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad += self.viewspace_point_list[idx].grad
                
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter], 
                    self.radii[self.visibility_filter]
                )
                
                self.gaussian.add_densification_stats(
                    viewspace_point_tensor_grad, 
                    self.visibility_filter
                )
                if self.true_global_step > 300 and self.true_global_step % 100 == 0:  # if self.true_global_step > self.opt.densify_from_iter and (self.true_global_step) % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.true_global_step > 500 else None # self.opt.opacity_reset_interval
                    # self.gaussian.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
                    self.gaussian.densify_and_prune(0.0002, 0.05, self.cameras_extent, size_threshold)
                
                # if self.true_global_step % self.opt.opacity_reset_interval == 0 or (self.bg_color == [0, 0, 0] and self.true_global_step == self.opt.densify_from_iter):
                #     self.gaussian.reset_opacity()
            else:
                if self.true_global_step % self.opt.mask_prune_iter == 0:
                    self.gaussian.mask_prune()

            print("ready for optimizer step")
            if self.true_global_step < self.opt.iterations:
                optimizer, optimizer_net = self.optimizers()
                scheduler_net = self.lr_schedulers()
                print("1")
                optimizer.step()
                print("2")
                optimizer.zero_grad(set_to_none = True)
                print("3")
                optimizer_net.step()
                print("4")
                optimizer_net.zero_grad(set_to_none = True)
                print("5")
                scheduler_net.step()
                print("6")

        return {"loss": loss}


    # def on_before_optimizer_step(self, optimizer):
    #     pass
        # print("on before optimizer step")
        # with torch.no_grad():

        #     if self.true_global_step < self.opt.densify_until_iter:
    
        #         viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
        #         for idx in range(len(self.viewspace_point_list)):
        #             viewspace_point_tensor_grad += self.viewspace_point_list[idx].grad
                
        #         self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
        #             self.gaussian.max_radii2D[self.visibility_filter], 
        #             self.radii[self.visibility_filter]
        #         )
                
        #         self.gaussian.add_densification_stats(
        #             viewspace_point_tensor_grad, 
        #             self.visibility_filter
        #         )

        #         if self.true_global_step > self.opt.densify_from_iter and (self.true_global_step) % self.opt.densification_interval == 0:
        #             size_threshold = 20 if self.true_global_step > self.opt.opacity_reset_interval else None
        #             self.gaussian.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
                
        #         # if self.true_global_step % self.opt.opacity_reset_interval == 0 or (self.bg_color == [0, 0, 0] and self.true_global_step == self.opt.densify_from_iter):
        #         #     self.gaussian.reset_opacity()
        #     else:
        #         if self.true_global_step % self.opt.mask_prune_iter == 0:
        #             self.gaussian.mask_prune()

        #     if self.true_global_step < self.opt.iterations:
        #         # optimizer, optimizer_net = self.optimizers()
        #         # scheduler_net = self.lr_schedulers()
        #         # optimizer.step()
        #         # optimizer.zero_grad(set_to_none = True)
        #         # optimizer_net.step()
        #         # optimizer_net.zero_grad(set_to_none = True)
        #         # scheduler_net.step()
        #         optimizer = self.optimizers()
        #         optimizer.step()
        #         optimizer.zero_grad(set_to_none = True)


    def on_train_end(self):
        pass
        # with torch.no_grad():
        #     storage = self.gaussian.final_prune(compress=False)
        #     with open(self.get_save_path("storage"), 'w') as c:
        #         c.write(storage)
        #     self.gaussian.precompute()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if False else [0, 0, 0]

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch,testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )


    def on_test_epoch_end(self):
        # 1. 将测试过程中生成的图像序列保存为视频
        self.save_img_sequence(
            f"it{self.true_global_step}-test",  # 输入路径
            f"it{self.true_global_step}-test",  # 输出路径
            "(\d+)\.png",                       # 文件名匹配模式
            save_format="mp4",                  # 保存为MP4格式
            fps=30,                             # 帧率30fps
            name="test",
            step=self.true_global_step,
        )
        
        # 2. 保存高斯点云模型为PLY文件
        save_path = self.get_save_path(f"last_3dgs.ply")
        self.gaussian.save_ply(save_path)
        
        # 3. 如果是从文本生成的形状（load_type=0）
        if self.load_type==0:
            # 保存原始点云
            o3d.io.write_point_cloud(
                self.get_save_path("shape.ply"), 
                self.point_cloud
            )
            # 保存shape-e生成的图像序列为GIF
            self.save_gif_to_file(
                self.shapeimages, 
                self.get_save_path("shape.gif")
            )
        
        # 4. 将PLY文件转换为带颜色的点云文件
        load_ply(
            save_path,
            self.get_save_path(f"it{self.true_global_step}-test-color.ply")
        )


    def configure_optimizers(self):
        # # 1. 创建参数解析器
        # self.parser = ArgumentParser(description="Training script parameters")

        # 2. 设置优化参数
        opt = OptimizationParams(self.parser)
        self.opt = opt
        
        # 3. 生成初始点云
        point_cloud = self.pcb()
        
        # 4. 设置相机范围
        self.cameras_extent = 4.0
        
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)

        # 6. 设置渲染管线参数
        self.pipe = PipelineParams(self.parser)
        
        # 7. 设置高斯模型的训练参数
        self.gaussian.training_setup(opt)
        
        # 8. 返回优化器配置
        # ret = {
        #     "optimizer": self.gaussian.optimizer,
        # }
        # return ret

        optimizer = self.gaussian.optimizer
        optimizer_net = self.gaussian.optimizer_net
        scheduler_net = self.gaussian.scheduler_net
        return (
            {"optimizer": optimizer},
            {"optimizer": optimizer_net, "lr_scheduler": scheduler_net},
        )
        # return [optimizer, optimizer_net], [scheduler_net]