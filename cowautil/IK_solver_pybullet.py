import pybullet as p
import pybullet_data
import numpy as np
import time


class robot_solver:
    def __init__(self, render=True):
        self.render = render
        # 1. 启动 pybullet 并加载机械臂模型
        if self.render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)  # 设置重力

        # 加载 URDF 机械臂模型
        self.ROBOT_ID = p.loadURDF('./assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf', [0.0, 0, 0.0], useFixedBase=True, physicsClientId=self.physicsClient)
        self.EE_LINK_INDEX = 19  # 作为EE的link的索引, 19对应l_ace #20 对于r_ace

        self.num_joints = p.getNumJoints(self.ROBOT_ID, physicsClientId=self.physicsClient)
        self.joint_info = [(p.getJointInfo(self.ROBOT_ID, i), i) for i in range(self.num_joints)]
        self.movable_joints = [info[1] for info in self.joint_info if p.getJointInfo(self.ROBOT_ID, info[1])[2] != p.JOINT_FIXED]

        self.movable_joint = [0., -0.8, 0.8, 0., 0., 0.]

        # 设置每个关节的初始角度
        for i, movable_joint in enumerate(self.movable_joints):
            p.resetJointState(self.ROBOT_ID, movable_joint, self.movable_joint[i], physicsClientId=self.physicsClient)

    def solve_fk(self, joint_rads):
        joint_positions = joint_rads
        for movable_joint_id, joint_position in zip(self.movable_joints, joint_positions):
            p.resetJointState(self.ROBOT_ID, movable_joint_id, joint_position, physicsClientId=self.physicsClient)
        # 获取机械臂ee link状态
        link_state = p.getLinkState(self.ROBOT_ID, self.EE_LINK_INDEX, physicsClientId=self.physicsClient)
        xyz_position = link_state[0]  # ee link的世界坐标系中的位置
        orientation = link_state[1]  # ee link的世界坐标系中的四元数姿态
        angle_axis = p.getAxisAngleFromQuaternion(orientation, physicsClientId=self.physicsClient) # 转轴角

        return np.array(xyz_position), np.array(angle_axis[0])*angle_axis[1]

    def quaternion_to_rotation_matrix(self,quaternion):
        '''
        Input quaternion [x y z w]
        Output rotation matrix R
        '''
        # 四元数到旋转矩阵的转换
        x,y,z,w= quaternion
        R = np.array([[1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]])
        return R

    def create_homogeneous_matrix(self, position, quaternion):
        # 创建4x4的齐次变换矩阵
        R = self.quaternion_to_rotation_matrix(quaternion)  # 旋转矩阵
        T = np.eye(4)  # 创建4x4单位矩阵
        T[:3, :3] = R  # 将旋转矩阵放入T的上左部分
        T[:3, 3] = position  # 将平移部分放入T的右上部分
        return T

    def solve_fk_homogeneous(self, joint_rads):
        '''
        Input joint_rads
        return the target_to_base_homogenious_matrix
        '''
        joint_positions = joint_rads
        for movable_joint_id, joint_position in zip(self.movable_joints, joint_positions):
            p.resetJointState(self.ROBOT_ID, movable_joint_id, joint_position, physicsClientId=self.physicsClient)
        # 获取机械臂ee link状态
        link_state = p.getLinkState(self.ROBOT_ID, self.EE_LINK_INDEX, physicsClientId=self.physicsClient)
        xyz_position = link_state[0]  # ee link的世界坐标系中的位置
        orientation = link_state[1]  # ee link的世界坐标系中的四元数姿态
        if self.render:
            for idx, movable_joint in enumerate(self.movable_joints):
                p.setJointMotorControl2(self.ROBOT_ID, movable_joint, p.POSITION_CONTROL, targetPosition=joint_positions[idx], physicsClientId=self.physicsClient)
            p.stepSimulation(physicsClientId=self.physicsClient)
            # for _ in range(500):
            #     p.stepSimulation()
            #     time.sleep(0.1)
        T = self.create_homogeneous_matrix(xyz_position, orientation)
        return T 

    # def solve_ik(self, target_pos_xyz: np.ndarray=[0.6, 0.0, 0.43], target_axis_of_rot: np.ndarray=[0, 0, 0]):
    def solve_ik(self, target_pos_xyz: np.ndarray=[0.98324077, 0.09237641, 0.55775578], target_axis_of_rot: np.ndarray=[-0.02569118192651871, 0.11636336587600085, 0.026930236058372834]):
        axis = np.array(target_axis_of_rot[:3])  # 提取旋转轴
        angle = np.linalg.norm(axis)  # 旋转角度（轴的模）
        if angle > 1e-6:  # 避免除以零
            axis_normalized = axis / angle  # 单位化旋转轴
        else:
            axis_normalized = [1, 0, 0]  # 默认值（无旋转时，旋转轴可选任意方向）
            angle = 0

        target_orientation = p.getQuaternionFromAxisAngle(axis_normalized, angle, physicsClientId=self.physicsClient)

        # 调用 inverse kinematics 函数
        joint_angles = p.calculateInverseKinematics(self.ROBOT_ID, self.EE_LINK_INDEX, target_pos_xyz, target_orientation, restPoses=self.movable_joint, physicsClientId=self.physicsClient)

        # 4. 设置可动关节的角度
        # 设置所有关节角度（如果关节是可动的）
        if self.render:
            for idx, movable_joint in enumerate(self.movable_joints):
                p.setJointMotorControl2(self.ROBOT_ID, movable_joint, p.POSITION_CONTROL, targetPosition=joint_angles[idx], physicsClientId=self.physicsClient)
            p.stepSimulation(physicsClientId=self.physicsClient)
            # for _ in range(500):
            #     p.stepSimulation()
            #     time.sleep(0.1)

        return joint_angles
    
    def solve_fk_velocity(self, joint_rad, joint_vel):
        """
        计算末端执行器在工具坐标系下的速度
        :param joint_rad: 关节角度 (弧度)
        :param joint_vel: 关节角速度
        :return: 末端执行器速度 (线速度 + 角速度) 在工具坐标系 (6x1)
        """
        # 计算正向运动学变换矩阵
        T = self.solve_fk_homogeneous(joint_rad)
        R = T[:3, :3]  # 旋转矩阵

        # 计算EE到 Base 变换矩阵
        T_EE_to_Base = np.eye(6)
        T_EE_to_Base[:3, :3] = R
        T_EE_to_Base[3:, 3:] = R

        # 计算 EE 到基座的逆变换
        T_Base_to_EE = np.linalg.inv(T_EE_to_Base)
        num_movable_joints = len(self.movable_joints)
        # 计算雅可比矩阵
        joint_accelerations = [0] * num_movable_joints # 关节加速度设为 0
        joint_velocity = [0] * num_movable_joints
        assert len(joint_rad) == num_movable_joints, f"joint_rad 长度错误，期望 {num_movable_joints}，但收到 {len(joint_rad)}"
        
        J_linear, J_angular = p.calculateJacobian(self.ROBOT_ID, self.EE_LINK_INDEX, [0, 0, 0], 
                                                joint_rad.tolist(), joint_velocity, joint_accelerations, physicsClientId=self.physicsClient)

        # 组合线速度和角速度雅可比矩阵
        J = np.vstack((np.array(J_linear), np.array(J_angular)))

        # 计算基座坐标系下的末端速度
        V_base_Coord = J @ np.array(joint_vel).reshape(-1,1)  # 确保维度匹配

        # 变换到工具坐标系
        # V_EE_Coord = T_Base_to_EE @ V_base_Coord

        return V_base_Coord
    def close(self):
        # 关闭仿真环境
        p.disconnect(physicsClientId=self.physicsClient)

if __name__ == "__main__":
    ik = robot_solver(render=True)
    t = time.time()
    # ik.solve_fk(joint_rads=[0.]*6)
    print(ik.solve_ik())
    print(time.time() - t)
