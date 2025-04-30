import pyrealsense2 as rs
import numpy as np
import cv2


class RealSenseTools:
    def __init__(self, cameral_serial="218622275838", w=848, h=480, fps=30, depth_max_distance=1):
        # Set param
        self.depth_max_distance = depth_max_distance # 大于此距离的像素在depth img中将被截断

        # Init RealSense
        self.pipeline = rs.pipeline()  # 定义流程pipeline

        config = rs.config()  # 定义配置config
        config.enable_device(cameral_serial)
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)  # 配置depth流
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)  # 配置color流

        self.profile = self.pipeline.start(config) 

        # 获取深度传感器的深度比例
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        align_to = rs.stream.color  # 对齐到color流
        self.align = rs.align(align_to)
        for _ in range(10):
            self.recv()
    
    def get_intrinsics(self):
        # 获取深度流和彩色流的内参
        depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()

        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()
        camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                                [0, color_intrinsics.fy, color_intrinsics.ppy],
                                [0, 0, 1]])

        # 获取畸变系数
        dist_coeffs = np.array(color_intrinsics.coeffs)

        # 保存到文件
        np.save("camera_matrix.npy", camera_matrix)
        np.save("dist_coeffs.npy", dist_coeffs)

        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)

        print({'fx': color_intrinsics.fx, 'fy': color_intrinsics.fy,
            'ppx': color_intrinsics.ppx, 'ppy': color_intrinsics.ppy,
            'height': color_intrinsics.height, 'width': color_intrinsics.width})

        return color_intrinsics, depth_intrinsics
    
    def recv_depth(self):
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = self.align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        depth_image_z16: np.array = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        depth_image_8bit: np.array = cv2.convertScaleAbs(depth_image_z16, alpha=255 * self.depth_scale/self.depth_max_distance)  # 深度图（8位）

        color_image_array: np.array = np.asanyarray(color_frame.get_data())  # RGB图

        # 伪彩色图
        depth_colormap_array: np.array = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)

        return color_image_array, depth_image_z16, aligned_depth_frame, depth_colormap_array
    
    def recv(self):
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = self.align.process(frames)  # 获取对齐帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
        color_image_array: np.array = np.asanyarray(color_frame.get_data())  # RGB图

        # 获取深度帧并进行处理
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        depth_image_z16: np.array = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        depth_image_8bit: np.array = cv2.convertScaleAbs(depth_image_z16, alpha=255 * self.depth_scale/self.depth_max_distance)  # 深度图（8位）

        # 伪彩色图
        depth_colormap_array: np.array = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)

        return color_image_array, depth_image_z16, depth_image_8bit, depth_colormap_array

        

# test 
if __name__ == "__main__":
    realsense = RealSenseTools()
    realsense.get_intrinsics()


    while True:
        color_img, depth_img, depth_frame, depth_colormap = realsense.recv()
        # cv2.imshow("color and depth", np.concatenate([color_img, depth_colormap], axis=1))
        cv2.imshow("color and depth", color_img)
        cv2.imshow("depth", depth_frame)
        print(depth_img)
        cv2.waitKey(1)

    