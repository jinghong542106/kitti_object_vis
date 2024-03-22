## 环境配置  
conda create -n kitti_vis python=3.8  
conda activate kitti_vis  

pip install opencv-python pillow scipy matplotlib
pip install open3d
## 主要功能：
实现的主要功能包括：  
● 显示原始相机图片  
● 显示原始点云数据  
● 显示带2D框的图片  
● 显示带3D框的图片  
● 显示带3D框的点云  
● 将点云投影到图片  
## 代码运行  
cd kitti_object_vis  
python  kitti_vision.py  
----------------------------------------   
1: show_origin_image  
2: show_origin_lidar  
3: show_image_with_2dbox   
4: show_image_with_project_3dbox  
5: show_lidar_with_3dbox  
6: show_image_with_lidar  
please choice number:6  

然后，依据需要选择  




