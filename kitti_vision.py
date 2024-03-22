import os
import cv2
import numpy as np
import open3d as o3d
from vision_utils import draw_lidar, draw_2dbox, gen_3dbox,project_box3d, draw_project, draw_box3d_lidar, get_lidar_in_image_fov


class Kitti:
    def __init__(self, root_path="./data/kitti/", ind=0) -> None:
        self.root_path = root_path
        train_file = os.path.join(root_path, "ImageSets/train.txt")
        with open(train_file, 'r') as f:
            names = f.readlines()
        self.names = [name.rstrip() for name in names]
        self.name = self.names[ind]

    def get_image(self, show=False):
        img_path = os.path.join(self.root_path, "training/image_2", self.name+".png")
        img = cv2.imread(img_path)
        if show and os.path.exists(img_path):
            cv2.imshow("origin image", img)
            if cv2.waitKey(0) == ord("q"):
                cv2.destroyAllWindows()
        return img

    def get_lidar(self, show=False):
        lidar_path = os.path.join(self.root_path, "training/velodyne", self.name+".bin")
        lidar = np.fromfile(lidar_path, dtype=np.float32)
        lidar = lidar.reshape((-1, 4))
        if show:
            #创建窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=800, height=600)
            # 创建坐标轴
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
            vis.add_geometry(mesh_frame)
            vis = draw_lidar(lidar,vis)
            vis.run()
        return lidar
    
    def get_calib(self):
        calib = {}
        calib_path = os.path.join(self.root_path, "training/calib", self.name+".txt")
        with open(calib_path, 'r') as cf:
            infos = cf.readlines()
            infos = [x.rstrip() for x in infos]
        for info in infos:
            if len(info) == 0:
                continue
            key, value = info.split(":", 1)
            calib[key] = np.array([float(x) for x in value.split()])
        calib_format = self.format_calib(calib)
        return calib_format

    def format_calib(self, calib):
        calib_format = {}
        # projection matrix from rect coord to image coord.
        rect2image = calib["P2"]
        rect2image = rect2image.reshape([3, 4])
        calib_format["rect2image"] = rect2image
        # projection matrix from lidar coord to reference cam coord.
        lidar2cam = calib["Tr_velo_to_cam"]
        lidar2cam = lidar2cam.reshape([3, 4])
        calib_format["lidar2cam"] = lidar2cam
        # projection matrix from rect cam coord to reference cam coord.
        rect2ref = calib["R0_rect"]
        rect2ref = rect2ref.reshape([3, 3])
        calib_format["rect2ref"] = rect2ref
        return calib_format

    def get_anns(self):
        anns = []
        label_path = os.path.join(self.root_path, "training/label_2", self.name+".txt")
        with open(label_path, 'r') as lf:
            labels = lf.readlines()
            labels = [label.rstrip() for label in labels]
        for label in labels:
            ann_format = {}
            ann = label.split(" ")
            class_name = ann[0]
            ann_format["class_name"]=class_name
            ann_ = [float(x) for x in ann[1:]]
            truncation = ann_[0] # truncated pixel ratio [0..1]
            ann_format["truncation"]=truncation
            occlusion = ann_[1] # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
            ann_format["occlusion"]=occlusion
            alpha = ann_[2]
            ann_format["alpha"]=alpha # object observation angle [-pi..pi]

            #2D box
            xmin, ymin, xmax, ymax = ann_[3], ann_[4], ann_[5], ann_[6]
            box2d = np.array([xmin, ymin, xmax, ymax])
            ann_format["box2d"]=box2d

            #3D box
            box3d = {}
            h, w, l = ann_[7], ann_[8], ann_[9]
            cx, cy, cz = ann_[10], ann_[11], ann_[12]
            box3d["dim"] = np.array([l, w, h])
            box3d["center"] = np.array([cx, cy, cz])
            yaw = ann_[13]
            box3d["rotation"] = yaw# yaw angle [-pi..pi]
            ann_format["box3d"]=box3d

            anns.append(ann_format)
        return anns

class VisKitti:
    def __init__(self, root_path="./data/kitti/", ind=0) -> None:
        self.kitti = Kitti(root_path=root_path, ind=ind)
        self.calib = self.kitti.get_calib()
        self.anns = self.kitti.get_anns()

    def show_origin_image(self):
        self.kitti.get_image(show=True)

    def show_origin_lidar(self):
        self.kitti.get_lidar(show=True)
        
    def show_image_with_2dbox(self, save=False):
        img = self.kitti.get_image()
        bbox = []
        names = []
        for ann in self.anns:
            bbox.append(ann["box2d"])
            names.append(ann["class_name"])
        draw_2dbox(img, bbox, names, save=save)
    
    def show_image_with_project_3dbox(self, show=True):
        img = self.kitti.get_image()
        bbox = []
        for ann in self.anns:
            bbox.append(ann["box3d"])
        bbox3d = gen_3dbox(bbox3d=bbox)
        project_xy,_ = project_box3d(bbox3d, self.calib)
        draw_project(img, project_xy, save=False)
    
    def show_lidar_with_3dbox(self, img_fov=False):
        bbox = []
        for ann in self.anns:
            bbox.append(ann["box3d"])
        bbox3d = gen_3dbox(bbox3d=bbox)
        #创建窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600)
        # 创建坐标轴
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
        vis.add_geometry(mesh_frame)
        
        lidar = self.kitti.get_lidar()
        vis = draw_lidar(lidar, vis)
        vis = draw_box3d_lidar(bbox3d, self.calib, vis)
        vis.run()
        
    def show_lidar_on_image(self,img_width=1238, img_height=374):
        """ Project LiDAR points to image """
        img = self.kitti.get_image()
        lidar = self.kitti.get_lidar()
        calib = self.calib
        
        imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
            lidar, calib, 0, 0, img_width, img_height, True)
        imgfov_pts_2d = pts_2d[fov_inds, :]
        
        import matplotlib
        cmap = matplotlib.colormaps["hsv"]
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        
        for i in range(imgfov_pts_2d.shape[0]):
            depth = abs(imgfov_pc_velo[i, 0])
            color = cmap[max(int(600.0 / depth),0), :]
            cv2.circle(
                img,
                (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
                2,
                color=tuple(color),
                thickness=-1,
            )
        cv2.imshow("projection", img)
        if cv2.waitKey(0) == ord("q"):
            cv2.destroyAllWindows()
            
        return img
            

if __name__ == "__main__":
    vis = VisKitti(ind=6)
    print("1: show_origin_image")
    print("2: show_origin_lidar")
    print("3: show_image_with_2dbox")
    print("4: show_image_with_project_3dbox")
    print("5: show_lidar_with_3dbox")
    print("6: show_image_with_lidar")
    
    choice = input("please choice number:")
    if choice=="1":
        vis.show_origin_image()
    elif choice=="2":
        vis.show_origin_lidar()
    elif choice=="3":
        vis.show_image_with_2dbox()
    elif choice=="4":
        vis.show_image_with_project_3dbox()
    elif choice=="5":
        vis.show_lidar_with_3dbox()
    elif choice=="6":
        vis.show_lidar_on_image()