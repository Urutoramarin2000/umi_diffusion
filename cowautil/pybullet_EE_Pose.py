import pybullet as p # 可以直接使用pip install pybullet安装
import pybullet_data
import numpy as np # 注意numpy版本不要高于2.0.0，可以使用pip install numpy==1.26.4安装
import zarr, os, time
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument('--urdf_path', '-u', type=str,
                    help='Path to the urdf', default='./assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf')
parser.add_argument('--dataset_path', '-d', type=str,
                    help='Path to the dataset', default="./20250113_120552.zarr")
parser.add_argument('--render', '-r', type=str,
                    help='Whether to render', default=False)
args = parser.parse_args()


if args.render:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置额外搜索路径

ROBOT_ID = p.loadURDF(args.urdf_path, [0, 0, 0], useFixedBase=True)
EE_LINK_INDEX = 19  # 作为EE的link的索引, 19对应l_ace


def get_joint_info():
    num_joints = p.getNumJoints(ROBOT_ID)
    joint_info_dict = {}

    for i in range(num_joints):
        info = p.getJointInfo(ROBOT_ID, i)
        joint_name = info[1].decode('UTF-8')  # 关节名称
        joint_type = info[2]  # 关节类型
        joint_info_dict[joint_name] = {
            'id': i,
            'type': joint_type,
            'info': info
        }

    return joint_info_dict


def get_link_info():
    num_joints = p.getNumJoints(ROBOT_ID)
    link_info_dict = {}

    for i in range(num_joints):
        info = p.getJointInfo(ROBOT_ID, i)
        joint_name = info[1].decode('UTF-8')  # 关节名称
        link_name = info[12].decode('UTF-8')  # 子link名称

        if link_name not in link_info_dict:
            link_info_dict[link_name] = {
                'id': i,
                'parent_joint_name': joint_name,
                'info': info
            }

    # 添加根link
    root_link_name = p.getBodyInfo(ROBOT_ID)[0].decode('UTF-8')
    link_info_dict[root_link_name] = {'id': -1, 'parent_joint_name': None}

    return link_info_dict


def get_movable_joints():
    num_joints = p.getNumJoints(ROBOT_ID)
    joint_info = [(p.getJointInfo(ROBOT_ID, i), i) for i in range(num_joints)]
    movable_joints = [info[1] for info in joint_info if p.getJointInfo(ROBOT_ID, info[1])[2] != p.JOINT_FIXED]
    return movable_joints


def init_ee_pos_vis():
    # 获取link状态
    link_state = p.getLinkState(ROBOT_ID, EE_LINK_INDEX)
    position = link_state[0]  # link的世界坐标系中的位置
    orientation = link_state[1]  # link的世界坐标系中的四元数姿态
    # 绘制坐标轴
    eepos_axis_x_line_id = p.addUserDebugLine(position, np.array(position) + 0.1 * np.array([1, 0, 0]), [1, 0, 0], lineWidth=5)  # X轴 (红色)
    eepos_axis_y_line_id = p.addUserDebugLine(position, np.array(position) + 0.1 * np.array([0, 1, 0]), [0, 1, 0], lineWidth=5)  # Y轴 (绿色)
    eepos_axis_z_line_id = p.addUserDebugLine(position, np.array(position) + 0.1 * np.array([0, 0, 1]), [0, 0, 1], lineWidth=5)  # Z轴 (蓝色)
    # 创建一个高亮显示的球体
    highlight_radius = 0.03
    highlight_color = [1, 0, 0, 1]  # 红色，带透明度
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=highlight_radius, rgbaColor=highlight_color)
    collision_shape_id = -1  # 不需要碰撞形状
    highlight_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)
    return highlight_body_id, eepos_axis_x_line_id, eepos_axis_y_line_id, eepos_axis_z_line_id


def update_debug_axis_lines(line_ids, length=0.3):
    # 获取机械臂ee link状态
    link_state = p.getLinkState(ROBOT_ID, EE_LINK_INDEX)
    position = link_state[0]  # ee link的世界坐标系中的位置
    orientation = link_state[1]  # ee link的世界坐标系中的四元数姿态
    # 解码四元数为旋转矩阵
    rotation_matrix = p.getMatrixFromQuaternion(orientation)
    rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

    # 定义坐标轴的方向向量
    x_axis = rotation_matrix @ np.array([length, 0, 0])
    y_axis = rotation_matrix @ np.array([0, length, 0])
    z_axis = rotation_matrix @ np.array([0, 0, length])

    # 更新X、Y、Z轴线段
    p.addUserDebugLine(position, position + x_axis, [1, 0, 0], lineWidth=5, replaceItemUniqueId=line_ids[0])  # X轴 (红色)
    p.addUserDebugLine(position, position + y_axis, [0, 1, 0], lineWidth=5, replaceItemUniqueId=line_ids[1])  # Y轴 (绿色)
    p.addUserDebugLine(position, position + z_axis, [0, 0, 1], lineWidth=5, replaceItemUniqueId=line_ids[2])  # Z轴 (蓝色)


def update_debug_highlight(highlight_body_id):
    # 获取机械臂ee link状态
    link_state = p.getLinkState(ROBOT_ID, EE_LINK_INDEX)
    position = link_state[0]  # ee link的世界坐标系中的位置
    orientation = link_state[1]  # ee link的世界坐标系中的四元数姿态
    # 更新高亮显示的球体
    p.resetBasePositionAndOrientation(highlight_body_id, position, orientation)


if __name__ == '__main__':
    print("-"*40)
    # 获取link信息字典
    link_info_dict = get_link_info()
    # 打印所有link信息
    for link_name, link_data in link_info_dict.items():
        print(f"Link Name: {link_name}, ID: {link_data['id']}")
    print("-"*40)

    print("-"*40)
    # 获取关节信息字典
    joint_info_dict = get_joint_info()
    # 打印所有关节信息
    for joint_name, joint_data in joint_info_dict.items():
        print(f"Joint Name: {joint_name}, ID: {joint_data['id']}, Type: {joint_data['type']}")
    print("-"*40)

    # 找到可移动关节（非固定关节）
    movable_joints = get_movable_joints()

    # read tejectory
    all_tejectory_joints_pos = zarr.open(os.path.join(args.dataset_path, "data", "joints_pos"), mode='r')
    # init highlight vis
    highlight_body_id, eepos_axis_x_line_id, eepos_axis_y_line_id, eepos_axis_z_line_id = init_ee_pos_vis()

    all_eef_pos = np.zeros((all_tejectory_joints_pos.shape[0], 3))
    all_eef_rot_axis_angle = np.zeros((all_tejectory_joints_pos.shape[0], 3))
    all_gripper_width = np.zeros((all_tejectory_joints_pos.shape[0], 1))
    # tejectory replay
    for time_step, joint in enumerate(all_tejectory_joints_pos):
        # 根据tejectory设置关节位置 fk
        joint_positions = joint[1:]
        for movable_joint_id, position in zip(movable_joints, joint_positions):
            p.resetJointState(ROBOT_ID, movable_joint_id, position)
        # 获取机械臂ee link状态
        link_state = p.getLinkState(ROBOT_ID, EE_LINK_INDEX)
        xyz_position = link_state[0]  # ee link的世界坐标系中的位置
        orientation = link_state[1]  # ee link的世界坐标系中的四元数姿态
        AxisAngle = p.getAxisAngleFromQuaternion(orientation)  # 四元数转rot axis

        all_eef_pos[time_step] = np.array(xyz_position)
        all_eef_rot_axis_angle[time_step] = np.array(AxisAngle[0]) * AxisAngle[1]
        all_gripper_width[time_step] = np.array(joint[:1])

        if args.render:
            update_debug_highlight(highlight_body_id)
            update_debug_axis_lines([eepos_axis_x_line_id, eepos_axis_y_line_id, eepos_axis_z_line_id])
            time.sleep(0.03)

        p.stepSimulation()

    # save ee_pos
    eef_pos = zarr.open(os.path.join(args.dataset_path, "data", "eef_pos"), mode='w', chunks=all_tejectory_joints_pos.chunks, shape=(all_tejectory_joints_pos.shape[0], 3), dtype=all_tejectory_joints_pos.dtype)
    eef_pos[:] = all_eef_pos

    eef_rot_axis_angle = zarr.open(os.path.join(args.dataset_path, "data", "eef_rot_axis_angle"), mode='w', chunks=all_tejectory_joints_pos.chunks, shape=(all_tejectory_joints_pos.shape[0], 3), dtype=all_tejectory_joints_pos.dtype)
    eef_rot_axis_angle[:] = all_eef_rot_axis_angle

    gripper_width = zarr.open(os.path.join(args.dataset_path, "data", "gripper_width"), mode='w', chunks=all_tejectory_joints_pos.chunks, shape=(all_tejectory_joints_pos.shape[0], 1), dtype=all_tejectory_joints_pos.dtype)
    gripper_width[:] = all_gripper_width
    
    exit(0)
