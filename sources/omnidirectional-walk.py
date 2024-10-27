# Omnidirectional Walking Pattern Generation for Humanoid Robot 
# By : Eko Rudiawan Jamzuri
# 31 December 2019
# This code is an implementation of paper
# Harada, Kensuke, et al. "An analytical method for real-time gait planning for humanoid robots." International Journal of Humanoid Robotics 3.01 (2006): 1-19.

import math
import numpy as np 
import scipy.io 
import itertools
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *
from pytransform3d.trajectories import *
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from pytransform3d.plot_utils import Trajectory
import matplotlib.animation as animation
from pytransform3d.transformations import transform_from, concat

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# import kinematics as kine

robot = RobotWrapper.BuildFromURDF("tes.urdf", root_joint=pin.JointModelFreeFlyer())
model = robot.model
data = robot.data



RIGHT_SUPPORT = 0
LEFT_SUPPORT = 1

class GaitController():
    def __init__(self):
        # Constant distance between hip to center 
        self.hip_offset = 0.035
        self.epsilon = np.finfo(np.float32).eps
        self.err_min = 1e-2
        # Konfigurasi link in mm
        self.CROTCH_TO_HIP = 0.035 # Jarak croth ke hip
        self.UPPER_HIP = 0.050 # Jarak hip yaw ke hip roll pitch
        self.HIP_TO_KNEE = 0.2215 # Panjang link upper leg
        self.KNEE_TO_ANKLE = 0.2215 # Panjang link lower leg
        self.ANKLE_TO_SOLE = 0.053 # Jarak ankle ke sole

        # Command for walking pattern
        # Defined as motion vector 
        self.cmd_x = 0.05
        self.cmd_y = 0.00
        self.cmd_a = np.radians(0)

        self.sx = 0.0
        self.sy = 0.0
        self.sa = 0.0
        
        # ZMP trajectory
        self.zmp_x = 0
        self.zmp_y = 0

        # Footsteps FIFO 
        # Use 3 foot pattern for 1 cycle gait
        self.footsteps = [[0.0,-self.hip_offset,0],
                          [0.0,self.hip_offset,0],
                          [0.0,-self.hip_offset,0]]

        self.zmp_x_record = []
        self.zmp_y_record = []

        self.footsteps_record = [[0.0,-self.hip_offset,0],
                                 [0.0,self.hip_offset,0],
                                 [0.0,-self.hip_offset,0]]

        # First support is right leg
        self.support_foot = RIGHT_SUPPORT

        # CoM pose
        self.com = [0,0,0,0,0,0,0]
        self.com_yaw = 0
        # Initial CoM yaw orientation
        self.init_com_yaw = 0.0 
        # Target CoM yaw orientation
        self.target_com_yaw = 0.0
        
        self.com_x_record = []
        self.com_y_record = []
        
        # Initial position and orientation for left swing foot
        self.init_lfoot_pose = np.zeros((7,1), dtype=float)
        self.init_lfoot_position = np.zeros((3,1), dtype=float)
        self.init_lfoot_orientation_yaw = 0.0 
        # Target position and orientation for left swing foot
        self.target_lfoot_pose = np.zeros((7,1), dtype=float)
        self.target_lfoot_position = np.zeros((3,1), dtype=float)
        self.target_lfoot_orientation_yaw = 0.0

        # Initial position and orientation for right swing foot
        self.init_rfoot_pose = np.zeros((7,1), dtype=float)
        self.init_rfoot_position = np.zeros((3,1), dtype=float)
        self.init_rfoot_orientation_yaw = 0.0
        # Target position and orientation for right swing foot
        self.target_rfoot_pose = np.zeros((7,1), dtype=float)
        self.target_rfoot_position = np.zeros((3,1), dtype=float)
        self.target_rfoot_orientation_yaw = 0.0
        
        # Current left foot and right foot pose written from world frame
        self.cur_rfoot = [0,-self.hip_offset,0,0,0,0,0]
        self.cur_lfoot = [0,self.hip_offset,0,0,0,0,0]

        # Current left foot and right foot pose written from CoM frame
        self.left_foot_pose = []
        self.right_foot_pose = []
        
        # Joint indices for quick access
        index = {name: i for i, name in enumerate(model.names)}
        print(index)
        index_dof = {name: model.joints[i].idx_q for i, name in enumerate(model.names)}
        # Get the initial positions of left and right feet
        left_foot_link = index["l_ank_pitch"]
        right_foot_link = index["r_ank_pitch"]
        # Perform forward kinematics for the initial configuration
        q0 = np.zeros(model.nq)  # Initial joint configuration
        pin.framesForwardKinematics(model, data, q0)  # Updates the frames' placements in `data`

        # Get the initial positions of left and right feet from the frames
        left_foot_pos0 = data.oMf[left_foot_link].translation
        right_foot_pos0 = data.oMf[right_foot_link].translation

        joint_angles = np.zeros(model.nq)
    
    def get_COM_height(self):
        # Define the configuration (e.g., zero configuration or a specific joint configuration)
        q = np.zeros(robot.model.nq)  # Adjust q as needed for your specific configuration

        # Calculate the COM position in world coordinates
        pin.centerOfMass(robot.model, robot.data, q)
        com_position = robot.data.com[0]  # COM in world coordinates

        # Perform forward kinematics to get the transformation of all frames
        pin.forwardKinematics(robot.model, robot.data, q)

        # Identify the frame index for the contact point on the ground
        foot_frame_name = "base_link"  # Replace with the actual foot frame name in your URDF
        foot_frame_id = robot.model.getFrameId(foot_frame_name)

        # Get the position of the foot frame (contact point) in world coordinates
        foot_position = robot.data.oMf[foot_frame_id].translation  # Foot position in world coordinates

        # Calculate the ZC parameter (height of COM from ground level)
        zc = com_position[2] - foot_position[2]  # Z-axis represents height
        
        return zc

        # print("COM Position from Ground:",  self.zc)

    # Set default gait parameter
    def get_gait_parameter(self):
        self.zc = self.get_COM_height() # CoM constant height
        self.max_swing_height = 0.03 # Maximum swing foot height 

        self.t_step = 0.25 # Timing for 1 cycle gait
        self.dsp_ratio = 0.15 # Percent of DSP phase
        self.dt = 0.01 # Control cycle

        self.t_dsp = self.dsp_ratio * self.t_step 
        self.t_ssp = (1.0 - self.dsp_ratio) * self.t_step
        self.t = 0
        self.dt_bez = 1 / (self.t_ssp / self.dt)
        self.t_bez = 0

    def print_gait_parameter(self):
        print("zc :", self.zc)
        print("dt :", self.dt)

    # Bezier curve function for generating rotation path
    def rot_path(self, init_angle, target_angle, time, t):
        p0 = np.array([[0],[init_angle]])
        p1 = np.array([[0],[target_angle]])
        p2 = np.array([[time],[target_angle]])
        p3 = np.array([[time],[target_angle]])
        path = np.power((1-t), 3)*p0 + 3*np.power((1-t), 2)*t*p1 + 3*(1-t)*np.power(t, 2)*p2 + np.power(t, 3)*p3
        return path

    # Bezier curve function for generating position path
    def swing_foot_path(self, str_pt, end_pt, swing_height, t):
        p0 = str_pt.copy()
        p1 = str_pt.copy()
        p1[2,0] = swing_height+(0.25*swing_height)
        p2 = end_pt.copy()
        p2[2,0] = swing_height+(0.25*swing_height)
        p3 = end_pt.copy()
        path = np.power((1-t), 3)*p0 + 3*np.power((1-t), 2)*t*p1 + 3*(1-t)*np.power(t, 2)*p2 + np.power(t, 3)*p3
        return path

    # Update support foot
    def swap_support_foot(self):
        if self.support_foot == RIGHT_SUPPORT:
            self.support_foot = LEFT_SUPPORT
        else:
            self.support_foot = RIGHT_SUPPORT

    # Function for generating swing foot trajectory
    # Result in foot pose written from world coordinate
    def get_foot_trajectory(self):
        # Get initial position and orientation of swing foot
        if self.t == 0:
            if self.support_foot == LEFT_SUPPORT:
                self.init_rfoot_pose[0,0] = self.cur_rfoot[0]
                self.init_rfoot_pose[1,0] = self.cur_rfoot[1]
                self.init_rfoot_pose[2,0] = 0
                self.init_rfoot_pose[3,0] = self.cur_rfoot[3]
                self.init_rfoot_pose[4,0] = self.cur_rfoot[4]
                self.init_rfoot_pose[5,0] = self.cur_rfoot[5]
                self.init_rfoot_pose[6,0] = self.cur_rfoot[6]
                # Set initial position of swing foot
                self.init_rfoot_position[0,0] = self.init_rfoot_pose[0,0]
                self.init_rfoot_position[1,0] = self.init_rfoot_pose[1,0]
                self.init_rfoot_position[2,0] = self.init_rfoot_pose[2,0]
                euler = euler_from_quaternion([self.init_rfoot_pose[3,0], self.init_rfoot_pose[4,0], self.init_rfoot_pose[5,0], self.init_rfoot_pose[6,0]])
                # Set initial yaw orientation from swing foot
                self.init_rfoot_orientation_yaw = euler[2] 

                # Set target foot pose from next footstep
                self.target_rfoot_pose[0,0] = self.footsteps[1][0]
                self.target_rfoot_pose[1,0] = self.footsteps[1][1]
                self.target_rfoot_pose[2,0] = 0
                q = quaternion_from_euler(0, 0, self.footsteps[1][2])
                self.target_rfoot_pose[3,0] = q[0]
                self.target_rfoot_pose[4,0] = q[1]
                self.target_rfoot_pose[5,0] = q[2]
                self.target_rfoot_pose[6,0] = q[3]
                # Set target position of swing foot
                self.target_rfoot_position[0,0] = self.target_rfoot_pose[0,0]
                self.target_rfoot_position[1,0] = self.target_rfoot_pose[1,0]
                self.target_rfoot_position[2,0] = self.target_rfoot_pose[2,0]
                euler = euler_from_quaternion([self.target_rfoot_pose[3,0], self.target_rfoot_pose[4,0], self.target_rfoot_pose[5,0], self.target_rfoot_pose[6,0]])
                # Set target orientation of swing foot
                self.target_rfoot_orientation_yaw = euler[2]
                euler = euler_from_quaternion([self.cur_lfoot[3], self.cur_lfoot[4], self.cur_lfoot[5], self.cur_lfoot[6]])
                support_foot_yaw = euler[2]
                # Calculate initial CoM yaw orientation and target CoM yaw orientation
                self.init_com_yaw = (support_foot_yaw + self.init_rfoot_orientation_yaw) / 2
                self.target_com_yaw = (support_foot_yaw + self.target_rfoot_orientation_yaw) / 2
            if self.support_foot == RIGHT_SUPPORT:
                self.init_lfoot_pose[0,0] = self.cur_lfoot[0]
                self.init_lfoot_pose[1,0] = self.cur_lfoot[1]
                self.init_lfoot_pose[2,0] = 0
                self.init_lfoot_pose[3,0] = self.cur_lfoot[3]
                self.init_lfoot_pose[4,0] = self.cur_lfoot[4]
                self.init_lfoot_pose[5,0] = self.cur_lfoot[5]
                self.init_lfoot_pose[6,0] = self.cur_lfoot[6]
                self.init_lfoot_position[0,0] = self.init_lfoot_pose[0,0]
                self.init_lfoot_position[1,0] = self.init_lfoot_pose[1,0]
                self.init_lfoot_position[2,0] = self.init_lfoot_pose[2,0]
                euler = euler_from_quaternion([self.init_lfoot_pose[3,0], self.init_lfoot_pose[4,0], self.init_lfoot_pose[5,0], self.init_lfoot_pose[6,0]])
                self.init_lfoot_orientation_yaw = euler[2]
                self.target_lfoot_pose[0,0] = self.footsteps[1][0]
                self.target_lfoot_pose[1,0] = self.footsteps[1][1]
                self.target_lfoot_pose[2,0] = 0
                q = quaternion_from_euler(0, 0, self.footsteps[1][2])
                self.target_lfoot_pose[3,0] = q[0]
                self.target_lfoot_pose[4,0] = q[1]
                self.target_lfoot_pose[5,0] = q[2]
                self.target_lfoot_pose[6,0] = q[3]
                self.target_lfoot_position[0,0] = self.target_lfoot_pose[0,0]
                self.target_lfoot_position[1,0] = self.target_lfoot_pose[1,0]
                self.target_lfoot_position[2,0] = self.target_lfoot_pose[2,0]
                euler = euler_from_quaternion([self.target_lfoot_pose[3,0], self.target_lfoot_pose[4,0], self.target_lfoot_pose[5,0], self.target_lfoot_pose[6,0]])
                self.target_lfoot_orientation_yaw = euler[2]
                euler = euler_from_quaternion([self.cur_rfoot[3], self.cur_rfoot[4], self.cur_rfoot[5], self.cur_rfoot[6]])
                support_foot_yaw = euler[2]
                self.init_com_yaw = (support_foot_yaw + self.init_lfoot_orientation_yaw) / 2
                self.target_com_yaw = (support_foot_yaw + self.target_lfoot_orientation_yaw) / 2

        # Generate foot trajectory 
        if self.t < (self.t_dsp/2.0) or self.t >= (self.t_dsp/2.0 + self.t_ssp):
            self.t_bez = 0
        else:
            if self.support_foot == LEFT_SUPPORT:
                self.cur_lfoot[0] = self.footsteps[0][0]
                self.cur_lfoot[1] = self.footsteps[0][1]
                self.cur_lfoot[2] = 0
                q = quaternion_from_euler(0,0,self.footsteps[0][2])
                self.cur_lfoot[3] = q[0]
                self.cur_lfoot[4] = q[1]
                self.cur_lfoot[5] = q[2]
                self.cur_lfoot[6] = q[3]
                path = self.swing_foot_path(self.init_rfoot_position, self.target_rfoot_position, self.max_swing_height, self.t_bez)
                self.cur_rfoot[0] = path[0,0]
                self.cur_rfoot[1] = path[1,0]
                self.cur_rfoot[2] = path[2,0]
                yaw_path = self.rot_path(self.init_rfoot_orientation_yaw, self.target_rfoot_orientation_yaw, self.t_ssp, self.t_bez)
                q = quaternion_from_euler(0,0,yaw_path[1,0])
                self.cur_rfoot[3] = q[0]
                self.cur_rfoot[4] = q[1]
                self.cur_rfoot[5] = q[2]
                self.cur_rfoot[6] = q[3]
            elif self.support_foot == RIGHT_SUPPORT:
                self.cur_rfoot[0] = self.footsteps[0][0]
                self.cur_rfoot[1] = self.footsteps[0][1]
                self.cur_rfoot[2] = 0
                q = quaternion_from_euler(0,0,self.footsteps[0][2])
                self.cur_rfoot[3] = q[0]
                self.cur_rfoot[4] = q[1]
                self.cur_rfoot[5] = q[2]
                self.cur_rfoot[6] = q[3]
                path = self.swing_foot_path(self.init_lfoot_position, self.target_lfoot_position, self.max_swing_height, self.t_bez)
                self.cur_lfoot[0] = path[0,0]
                self.cur_lfoot[1] = path[1,0]
                self.cur_lfoot[2] = path[2,0]
                yaw_path = self.rot_path(self.init_lfoot_orientation_yaw, self.target_lfoot_orientation_yaw, self.t_ssp, self.t_bez)
                q = quaternion_from_euler(0,0, yaw_path[1,0])
                self.cur_lfoot[3] = q[0]
                self.cur_lfoot[4] = q[1]
                self.cur_lfoot[5] = q[2]
                self.cur_lfoot[6] = q[3]

            # Generate CoM yaw path
            yaw_path = self.rot_path(self.init_com_yaw, self.target_com_yaw, self.t_ssp, self.t_bez)
            self.com_yaw = yaw_path[1,0]
            self.t_bez += self.dt_bez
    
    # Function for generating zmp trajectory
    def get_zmp_trajectory(self):
        epsilon = 0.0001 
        td = self.t % self.t_step 
        if td > -epsilon and td < epsilon:
            self.t0 = self.t
            self.t1 = self.t0 + (self.t_ssp / 2)
            self.t2 = self.t1 + self.t_dsp
            self.tf = self.t_step
            # Initial CoM position
            self.com0_x = self.footsteps[0][0] + (self.footsteps[1][0] - self.footsteps[0][0]) / 2
            self.com0_y = self.footsteps[0][1] + (self.footsteps[1][1] - self.footsteps[0][1]) / 2
            # Final CoM position
            self.com1_x = self.footsteps[1][0] + (self.footsteps[2][0] - self.footsteps[1][0]) / 2
            self.com1_y = self.footsteps[1][1] + (self.footsteps[2][1] - self.footsteps[1][1]) / 2
            # Support foot
            self.sup_x = self.footsteps[1][0]
            self.sup_y = self.footsteps[1][1]

        if self.t >= self.t0 and self.t < self.t1:
            self.zmp_x = self.com0_x+((self.sup_x-self.com0_x)/(self.t1-self.t0))*self.t
            self.zmp_y = self.com0_y+((self.sup_y-self.com0_y)/(self.t1-self.t0))*self.t
        elif self.t >= self.t1 and self.t < self.t2:
            self.zmp_x = self.sup_x
            self.zmp_y = self.sup_y 
        elif self.t >= self.t2 and self.t < self.tf:
            self.zmp_x=self.sup_x+((self.com1_x-self.sup_x)/(self.tf-self.t2))*(self.t-self.t2)
            self.zmp_y=self.sup_y+((self.com1_y-self.sup_y)/(self.tf-self.t2))*(self.t-self.t2)
        self.zmp_x_record.append(self.zmp_x)
        self.zmp_y_record.append(self.zmp_y)
    
    # Add new footstep to FIFO buffer
    def add_new_footstep(self):
        self.footsteps.pop(0)
        if self.support_foot == LEFT_SUPPORT: 
            self.sx = self.cmd_x
            self.sy = -2*self.hip_offset + self.cmd_y
            self.sa += self.cmd_a
            dx = self.footsteps[-1][0] + np.cos(self.sa) * self.sx + (-np.sin(self.sa) * self.sy)
            dy = self.footsteps[-1][1] + np.sin(self.sa) * self.sx + np.cos(self.sa) * self.sy
            self.footsteps.append([dx, dy, self.sa])
            self.footsteps_record .append([dx, dy, self.sa])
        elif self.support_foot == RIGHT_SUPPORT:
            self.sx = self.cmd_x 
            self.sy = 2*self.hip_offset + self.cmd_y
            self.sa += self.cmd_a
            dx = self.footsteps[-1][0] + np.cos(self.sa) * self.sx + (-np.sin(self.sa) * self.sy)
            dy = self.footsteps[-1][1] + np.sin(self.sa) * self.sx + np.cos(self.sa) * self.sy
            self.footsteps.append([dx, dy, self.sa])
            self.footsteps_record .append([dx, dy, self.sa])
        self.swap_support_foot()

    # Function for generating CoM trajectory
    def get_com_trajectory(self):
        self.Tc = np.sqrt(9.81/self.zc)
        cx = np.array([0,
                       (np.sinh(self.Tc*(self.t1 - self.tf))*(self.sup_x - self.com0_x))/(self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - (np.sinh(self.Tc*(self.t2 - self.tf))*(self.sup_x - self.com1_x))/(self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.cosh(self.Tc*(2*self.t1 - self.tf)) - np.cosh(self.Tc*self.tf))*(self.sup_x - self.com0_x))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.cosh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.cosh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_x - self.com1_x))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.sinh(self.Tc*(2*self.t1 - self.tf)) + np.sinh(self.Tc*self.tf))*(self.sup_x - self.com0_x))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.sinh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.sinh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_x - self.com1_x))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.cosh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.cosh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_x - self.com0_x))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.cosh(self.Tc*(2*self.t2 - self.tf)) - np.cosh(self.Tc*self.tf))*(self.sup_x - self.com1_x))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((self.sup_x - self.com0_x)*(np.sinh(self.Tc*(self.t1 + self.t2 - self.tf)) + np.sinh(self.Tc*(self.t1 - self.t2 + self.tf))))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.sinh(self.Tc*(2*self.t2 - self.tf)) + np.sinh(self.Tc*self.tf))*(self.sup_x - self.com1_x))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf))])

        cy = np.array([0,
                       (np.sinh(self.Tc*(self.t1 - self.tf))*(self.sup_y - self.com0_y))/(self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - (np.sinh(self.Tc*(self.t2 - self.tf))*(self.sup_y - self.com1_y))/(self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.cosh(self.Tc*(2*self.t1 - self.tf)) - np.cosh(self.Tc*self.tf))*(self.sup_y - self.com0_y))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.cosh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.cosh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_y - self.com1_y))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.sinh(self.Tc*(2*self.t1 - self.tf)) + np.sinh(self.Tc*self.tf))*(self.sup_y - self.com0_y))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.sinh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.sinh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_y - self.com1_y))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.cosh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.cosh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_y - self.com0_y))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.cosh(self.Tc*(2*self.t2 - self.tf)) - np.cosh(self.Tc*self.tf))*(self.sup_y - self.com1_y))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((self.sup_y - self.com0_y)*(np.sinh(self.Tc*(self.t1 + self.t2 - self.tf)) + np.sinh(self.Tc*(self.t1 - self.t2 + self.tf))))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.sinh(self.Tc*(2*self.t2 - self.tf)) + np.sinh(self.Tc*self.tf))*(self.sup_y - self.com1_y))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf))])

        if self.t >= self.t0 and self.t < self.t1:
            self.com[0] = self.com0_x+((self.sup_x-self.com0_x)/(self.t1-self.t0))*(self.t-self.t0)+cx[0]*np.cosh(self.Tc*self.t)+cx[1]*np.sinh(self.Tc*self.t)
            self.com[1] = self.com0_y+((self.sup_y-self.com0_y)/(self.t1-self.t0))*(self.t-self.t0)+cy[0]*np.cosh(self.Tc*self.t)+cy[1]*np.sinh(self.Tc*self.t)
        elif self.t >= self.t1 and self.t < self.t2:
            self.com[0] = self.sup_x+cx[2]*np.cosh(self.Tc*(self.t-self.t1))+cx[3]*np.sinh(self.Tc*(self.t-self.t1))
            self.com[1] = self.sup_y+cy[2]*np.cosh(self.Tc*(self.t-self.t1))+cy[3]*np.sinh(self.Tc*(self.t-self.t1))
        elif self.t >= self.t2 and self.t < self.tf:
            self.com[0] = self.sup_x+((self.com1_x-self.sup_x)/(self.tf-self.t2))*(self.t-self.t2)+cx[4]*np.cosh(self.Tc*(self.t-self.t2))+cx[5]*np.sinh(self.Tc*(self.t-self.t2))
            self.com[1] = self.sup_y+((self.com1_y-self.sup_y)/(self.tf-self.t2))*(self.t-self.t2)+cy[4]*np.cosh(self.Tc*(self.t-self.t2))+cy[5]*np.sinh(self.Tc*(self.t-self.t2))
        # CoM height is constant
        self.com[2] = self.zc 
        # CoM orientation 
        q = quaternion_from_euler(0, 0, self.com_yaw)
        self.com[3] = q[0]
        self.com[4] = q[1]
        self.com[5] = q[2]
        self.com[6] = q[3]
        self.com_x_record.append(self.com[0])
        self.com_y_record.append(self.com[1])

    # Create transformation matrix
    def create_tf_matrix(self, list_xyz_qxyzw):
        T_mat = np.eye(4)
        T_mat[0,3] = list_xyz_qxyzw[0]
        T_mat[1,3] = list_xyz_qxyzw[1]
        T_mat[2,3] = list_xyz_qxyzw[2]
        R_mat = matrix_from_quaternion([list_xyz_qxyzw[6], list_xyz_qxyzw[3], list_xyz_qxyzw[4], list_xyz_qxyzw[5]])
        T_mat[:3,:3] = R_mat
        return T_mat

    # Function for tranform left foot and right foot pose into CoM frame
    def get_foot_pose(self):
        world_to_com = self.create_tf_matrix(self.com)
        world_to_lfoot = self.create_tf_matrix(self.cur_lfoot)
        world_to_rfoot = self.create_tf_matrix(self.cur_rfoot)
        world_to_com_inv = np.linalg.pinv(world_to_com)
        com_to_lfoot = world_to_com_inv.dot(world_to_lfoot)
        com_to_rfoot = world_to_com_inv.dot(world_to_rfoot)
        q_lfoot = quaternion_from_matrix(com_to_lfoot[:3,:3])
        q_rfoot = quaternion_from_matrix(com_to_rfoot[:3,:3])
        self.left_foot_pose = [com_to_lfoot[0,3], com_to_lfoot[1,3], com_to_lfoot[2,3], q_lfoot[1], q_lfoot[2], q_lfoot[3], q_lfoot[0]]
        self.right_foot_pose = [com_to_rfoot[0,3], com_to_rfoot[1,3], com_to_rfoot[2,3], q_rfoot[1], q_rfoot[2], q_rfoot[3], q_rfoot[0]]
    
    # Function for getting walking pattern
    def get_walking_pattern(self):
        self.get_zmp_trajectory()
        self.get_com_trajectory()
        self.get_foot_trajectory()
        self.get_foot_pose()
        self.t += self.dt 
        if self.t > self.t_step:
            self.t = 0
            self.add_new_footstep()

    def initialize(self):
        self.get_gait_parameter()
        self.print_gait_parameter()
    
    def print_pose(self, name, pose, rpy_mode = True):
        if rpy_mode:
            euler = euler_from_quaternion([pose[3], pose[4], pose[5], pose[6]])
            print(name + " xyz : " + "{0:.3f}".format(pose[0]) +", "+ "{0:.3f}".format(pose[1]) +", "+ "{0:.3f}".format(pose[2]) + \
            " rpy : " + "{0:.3f}".format(euler[0]) +", "+ "{0:.3f}".format(euler[1]) +", "+ "{0:.3f}".format(euler[2]))
        else:
            print(name + " xyz : " + "{0:.3f}".format(pose[0]) +", "+ "{0:.3f}".format(pose[1]) +", "+ "{0:.3f}".format(pose[2]) + \
            " xyzw : " + "{0:.3f}".format(pose[3]) +", "+ "{0:.3f}".format(pose[4]) +", "+ "{0:.3f}".format(pose[5]) +", "+ "{0:.3f}".format(pose[6]))
        
    def end(self):
        pass
    
    def pose_error(self, f_target, f_result):
        f_diff = f_target.Inverse() * f_result
        [dx, dy, dz] = f_diff.p
        [drz, dry, drx] = f_diff.M.GetEulerZYX()
        error = np.sqrt(dx**2 + dy**2 + dz**2 + drx**2 + dry**2 + drz**2)
        error_list = [dx, dy, dz, drx, dry, drz]
        return error, error_list

    def calculate_leg_kinematics(self, x, y, z, yaw, CROTCH_TO_HIP, UPPER_HIP, HIP_TO_KNEE, KNEE_TO_ANKLE, ANKLE_TO_SOLE, invert=False):
        x_from_hip = (x - 0)
        y_from_hip = (y - CROTCH_TO_HIP)
        z_from_hip = (z + (UPPER_HIP + ANKLE_TO_SOLE))

        xa = x_from_hip
        ya = xa * np.tan(yaw)
        beta = np.pi/2 - yaw
        yb = (y_from_hip - ya)
        gamma = np.pi/2 - beta
        xb = xa / np.cos(yaw) + np.sin(gamma) * (y_from_hip - ya)
        x_from_hip_yaw = xb
        y_from_hip_yaw = yb
        z_from_hip_yaw = z_from_hip

        C = np.sqrt(xb**2 + yb**2 + z_from_hip_yaw**2)
        zb = np.sqrt(yb**2 + z_from_hip_yaw**2)
        zc = np.sqrt(xb**2 + z_from_hip_yaw**2)
        zeta = np.arctan2(yb, zc)
        Cb = np.sign(xb)*np.sqrt(C**2 - zb**2)

        q_hip_yaw = yaw
        q_hip_roll = zeta 
        q_hip_pitch = -(np.arctan2(Cb, np.sign(z_from_hip_yaw)*z_from_hip_yaw) + np.arccos((C/2)/HIP_TO_KNEE))
        q_knee = np.pi-(2*(np.arcsin((C/2)/HIP_TO_KNEE)))
        q_ankle_pitch = -(np.pi/2 - (np.arctan2(np.sign(z_from_hip_yaw)*z_from_hip_yaw, Cb) + np.arccos((C/2)/HIP_TO_KNEE)))
        q_ankle_roll = -q_hip_roll

        if invert:
            q_hip_yaw = -q_hip_yaw
            q_hip_roll = -q_hip_roll
            q_hip_pitch = -q_hip_pitch
            q_ankle_pitch = -q_ankle_pitch
            q_ankle_roll = -q_ankle_roll
        
        print("hip_yaw :", (q_hip_yaw))
        print("q_hip_roll", q_hip_roll)
        print("hip_roll :", (q_hip_roll))
        print("cb ", Cb)
        print("hip_pitch :", (q_hip_pitch))
        print("knee :", (q_knee))
        print("ankle_pitch :", (q_ankle_pitch))
        print("ankle_roll :", (q_ankle_roll))
        
        print("=====================================")

        return q_hip_yaw, q_hip_roll, q_hip_pitch, q_knee, q_ankle_pitch, q_ankle_roll
    
    # Used for Inverse Kinematic
    # init_pos and target_pos is servo angle calculated by inverse kinematic
    # Return a smooth trajectory
    def minimum_jerk_trajectory(self, init_pos, target_pos, total_time=0.5, dt=0.01):
        xi = init_pos
        xf = target_pos
        d = total_time
        list_t = []
        list_x = []
        t = 0
        while t < d:
            x = xi + (xf-xi) * (10*(t/d)**3 - 15*(t/d)**4 + 6*(t/d)**5)
            list_t.append(t)
            list_x.append(x)
            t += dt
        return np.array(list_t), np.array(list_x)

    def run(self):
        print("===========================")
        print("Gait Controller   ")
        print("===========================")
        self.initialize()
        t_sim = 5
        t = 0
        com_trajectory = []
        lfoot_trajectory = []
        rfoot_trajectory = []
        while t < t_sim:
            self.get_walking_pattern()
            com_trajectory.append([self.com[0], self.com[1], self.com[2], self.com[6], self.com[3], self.com[4], self.com[5]])
            rfoot_trajectory.append([self.cur_rfoot[0], self.cur_rfoot[1], self.cur_rfoot[2], self.cur_rfoot[6], self.cur_rfoot[3], self.cur_rfoot[4], self.cur_rfoot[5]])
            lfoot_trajectory.append([self.cur_lfoot[0], self.cur_lfoot[1], self.cur_lfoot[2], self.cur_lfoot[6], self.cur_lfoot[3], self.cur_lfoot[4], self.cur_lfoot[5]])
            self.print_pose("com", self.com, rpy_mode=False)
            self.print_pose("left foot pose", self.left_foot_pose, rpy_mode=False)
            self.print_pose("right foot pose", self.right_foot_pose, rpy_mode=False)
            print("=============================================")
            
            t += self.dt


        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        com_trajectory = np.array(com_trajectory)
        rfoot_trajectory = np.array(rfoot_trajectory)
        lfoot_trajectory = np.array(lfoot_trajectory)
        plot_trajectory(ax=ax, P=com_trajectory, s=0.01, show_direction=False)
        plot_trajectory(ax=ax, P=rfoot_trajectory, s=0.01, show_direction=False)
        plot_trajectory(ax=ax, P=lfoot_trajectory, s=0.01, show_direction=False)
        ax.set_ylim(-0.2,0.7)
        ax.set_zlim(0.0,0.5)
        plt.show()
        self.end()

def main():
    gc = GaitController()
    gc.run()

if __name__ == "__main__":
    main()