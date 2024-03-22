import os
import cv2
import numpy as np
import open3d as o3d


def draw_lidar(pc,vis):
    points=pc[:,:3]
    points_intensity = pc[:, 3] *255 # intensity 
    #将array格式点云转为open3d
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    # pointcloud.paint_uniform_color([1,0,0])
    
    points_colors =np.zeros([points.shape[0],3])
    for i in range(points_intensity.shape[0]):
        points_colors[i,:] =[1, points_intensity[i], points_intensity[i]*0.5]
    pointcloud.colors = o3d.utility.Vector3dVector(points_colors)  # 根据 intensity 为点云着色
    
    # 设置点云渲染参数
    opt=vis.get_render_option()
    # 设置背景色（这里为白色）
    opt.background_color=np.array([255, 255, 255])
    # 设置渲染点的大小
    opt.point_size=2.0
    # 添加点云
    vis.add_geometry(pointcloud)
    return vis

def draw_gt_boxes3d(gt_boxes3d, vis):
    """ Draw 3D bounding boxes, gt_boxes3d: XYZs of the box corners"""
    num = len(gt_boxes3d)
    for n in range(num):
        points_3dbox = gt_boxes3d[n]
        lines_box = np.array([[0, 1], [1, 2], [2, 3],[0, 3], [4, 5], [5, 6], [6, 7], [4, 7], 
                        [0, 4], [1, 5], [2, 6], [3, 7]]) #指明哪两个顶点之间相连
        colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
        line_set = o3d.geometry.LineSet() #创建line对象
        line_set.lines = o3d.utility.Vector2iVector(lines_box) #将八个顶点转换成o3d可以使用的数据类型
        line_set.colors = o3d.utility.Vector3dVector(colors)  #设置每条线段的颜色
        line_set.points = o3d.utility.Vector3dVector(points_3dbox) #把八个顶点的空间信息转换成o3d可以使用的数据类型
        #将矩形框加入到窗口中
        vis.add_geometry(line_set) 
    return vis

def draw_box3d_lidar(bbox3d, calib, vis):
    # method 1
    lidar2cam = calib["lidar2cam"]
    lidar2cam = expand_matrix(lidar2cam)
    cam2rect_ = calib["rect2ref"]
    cam2rect = np.eye(4, 4)
    cam2rect[:3, :3] = cam2rect_
    lidar2rec = np.dot(lidar2cam, cam2rect)
    rec2lidar = np.linalg.inv(lidar2rec) #(AB)-1 = B-1@A-1
    
    all_lidar_box3d = []
    for box3d in bbox3d:
        if np.any(box3d[2, :] < 0.1):
            continue
        box3d = np.concatenate([box3d, np.ones((1, 8))], axis=0)
        lidar_box3d = np.dot(rec2lidar, box3d)[:3, :]
        lidar_box3d = np.transpose(lidar_box3d)
        all_lidar_box3d.append(lidar_box3d)
    vis = draw_gt_boxes3d(all_lidar_box3d, vis)
    return vis

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=1.0):
    """ Filter lidar points, keep those in image FOV """
    lidar2cam = calib["lidar2cam"]
    lidar2cam = expand_matrix(lidar2cam)
    cam2rect_ = calib["rect2ref"]
    cam2rect = np.eye(4, 4)
    cam2rect[:3, :3] = cam2rect_
    lidar2rec = np.dot(cam2rect, lidar2cam)
    P = calib["rect2image"]
    P = expand_matrix(P)
    project_velo_to_image = np.dot(P, lidar2rec)
    
    pc_velo_T = pc_velo.T
    pc_velo_T = np.concatenate([pc_velo_T[:3,:], np.ones((1, pc_velo_T.shape[1]))], axis=0)
    
    project_3dbox = np.dot(project_velo_to_image, pc_velo_T)[:3, :]
    pz = project_3dbox[2, :]
    px = project_3dbox[0, :]/pz
    py = project_3dbox[1, :]/pz
    pts_2d = np.vstack((px, py)).T
    
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]

    return imgfov_pc_velo, pts_2d, fov_inds
    
def draw_2dbox(img, bbox, names=None, save=False):
    assert len(bbox)==len(names), "names not match bbox"
    color_map = {"Car":(0, 255, 0), "Pedestrian":(255, 0, 0), "Cyclist":(0, 0, 255)}
    for i, box in enumerate(bbox):
        name = names[i]
        if name not in color_map.keys():
            continue
        color = color_map[name]
        cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            2,
        )
        name_coord = (int(box[0]), int(max(box[1]-5, 0)))
        cv2.putText(img, name, name_coord, 
                    cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    cv2.imshow("image_with_2dbox", img)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite("image_with_2dbox.jpg", img)

def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def expand_matrix(matrix):
    new_matrix = np.eye(4, 4)
    new_matrix[:3, :] = matrix
    return new_matrix

def gen_3dbox(bbox3d):
    corners_3d_all = []
    for box in bbox3d:
        center = box["center"]
        l, w, h = box["dim"]
        angle = box["rotation"]
        R = roty(angle)
        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = np.dot(R, corners)
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        corners_3d_all.append(corners_3d)
    return corners_3d_all



def project_box3d(bbox3d, calib):
    P = calib["rect2image"]
    P = expand_matrix(P)
    project_xy = []
    project_z = []
    for box3d in bbox3d:
        if np.any(box3d[2, :] < 0.1):
            continue
        box3d = np.concatenate([box3d, np.zeros((1, 8))], axis=0)
        project_3dbox = np.dot(P, box3d)[:3, :]
        pz = project_3dbox[2, :]
        px = project_3dbox[0, :]/pz
        py = project_3dbox[1, :]/pz
        xy = np.stack([px, py], axis=1)
        project_xy.append(xy)
        project_z.append(pz)
    print(project_xy)    
    return project_xy, project_z

def draw_project(img, project_xy, save=False):
    color_map = {"Car":(0, 255, 0), "Pedestrian":(255, 0, 0), "Cyclist":(0, 0, 255)}
    for i, qs in enumerate(project_xy):
        color = (0, 255, 0)
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            # use LINE_AA for opencv3
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, 1)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, 1)
            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, 1)

    cv2.imshow("image_with_projectbox", img)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite("image_with_projectbox.jpg", img)


