# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements a map to a 3D point on the robot body.
"""

import os
import torch

import warp as wp
import warp.torch

from fabrics_sim.prod.kinematics import Kinematics
from fabrics_sim.taskmaps.maps_base import BaseMap

# Define PyTorch autograd op to wrap foward kinematics
# function.
class RobotKinematics(torch.autograd.Function):

    @staticmethod
    def forward(ctx, joint_q, robot_kinematics):

        # Hold onto recording of kernel launches.
        ctx.tape = wp.Tape()

        # Hold onto inputs and outputs
        ctx.joint_q = wp.torch.from_torch(joint_q)
        ctx.robot_kinematics = robot_kinematics
        
        with ctx.tape:
            ctx.robot_kinematics.eval(ctx.joint_q, jacobians=True)
            #ctx.robot_kinematics.eval(ctx.joint_q, batch_qd=ctx.joint_q, velocities=True, jacobians=True)
        
        return (wp.torch.to_torch(ctx.robot_kinematics.batch_link_transforms),
                wp.torch.to_torch(ctx.robot_kinematics.batch_link_jacobians))

    @staticmethod
    def backward(ctx, adj_link_transforms, adj_jacobians):

        # Map incoming Torch grads to our output variables
        grads = { ctx.robot_kinematics.batch_link_transforms:
                      wp.torch.from_torch(adj_link_transforms, dtype=wp.transform),
                  ctx.robot_kinematics.batch_link_jacobians:
                      wp.torch.from_torch(adj_jacobians, dtype=wp.vec3) }

        # Calculate gradients
        ctx.tape.zero()
        ctx.tape.backward(grads=grads)

        # Return adjoint w.r.t. inputs
        return (wp.torch.to_torch(ctx.tape.gradients[ctx.joint_q]),
                None,
                None)

class RobotFrameOriginsTaskMap(BaseMap):
    def __init__(self, urdf_path, link_names, batch_size, device, active_indices=None, mimic_info=None):
        """
        Constructor for building the desired robot taskmap.
        -----------------------------------------
        :param urdf_path: str, robot URDF filepath
        :param link_names: list of link names (str) of the robot to build the taskmap
        :param batch_size: int, size of the batch of robots
        :param device: type str that sets the cuda device for the fabric
        :param active_indices: list of int, indices of active joints in the full configuration vector
        :param mimic_info: dict, mapping from mimic joint index to (parent joint index, multiplier, offset)
        """
        super().__init__(device)

        # Allocate for robot kinemtics, the relevant link indices, and the batch size.
        self.urdf_path = urdf_path
        self.robot_kinematics = None
        self.link_names = link_names
        self.link_indices = None
        self.batch_size = batch_size
        self.active_indices = active_indices
        self.mimic_info = mimic_info

        self.init_robot_kinematics(self.batch_size)

    def init_robot_kinematics(self, batch_size):
        # Create the robot kinematics object that wraps several Warp kernels for computing
        # forward kinematics
        multithreading = False
        self.robot_kinematics = Kinematics(self.urdf_path, batch_size, multithreading,
                                           device=self.device)

        self.link_indices =  []
        for link_name in self.link_names:
            self.link_indices.append(self.robot_kinematics.get_link_index(link_name))
        self.link_indices = torch.tensor(self.link_indices, device=self.device)

        self.batch_size = batch_size

    def forward_position(self, q, features):
        # Check if the batch size matches the batch size of the incoming q. If not,
        # then re-initialize the robots kinematics.
#        if self.batch_size != q.shape[0]:
#            self.init_robot_kinematics(q.shape[0])

        # Handle mimic joints if specified
        if self.active_indices is not None and self.mimic_info is not None:
            # Construct full q
            # Determine full dimension
            # Assuming max index in active_indices and mimic_info keys covers all joints
            max_idx = max(max(self.active_indices), max(self.mimic_info.keys()) if self.mimic_info else 0)
            full_dim = max_idx + 1
            
            q_full = torch.zeros(self.batch_size, full_dim, device=self.device)
            
            # Fill active joints
            # q is (batch, num_active)
            for i, idx in enumerate(self.active_indices):
                q_full[:, idx] = q[:, i]
                
            # Fill mimic joints
            for mimic_idx, (parent_idx, multiplier, offset) in self.mimic_info.items():
                q_full[:, mimic_idx] = q_full[:, parent_idx] * multiplier + offset
                
            q_input = q_full
        else:
            q_input = q

        # Calculate the link transforms and their origin Jacobians.
        link_transforms, jacobians = RobotKinematics.apply(q_input, self.robot_kinematics)

        # Pull out the position of the origins and stack them across all desired frames.
        x = link_transforms[:, self.link_indices, :3].reshape((self.batch_size,
                                                               len(self.link_indices) * 3))

        # Pull out the Jacobians and stack them for the desired frames.
        # jacobian is of shape (batch_size, num_links, root_dim, 3)
        
        if self.active_indices is not None and self.mimic_info is not None:
            # We need to condense the jacobian from full_dim to num_active
            # J_parent_new = J_parent_old + J_mimic * multiplier
            
            # First, copy the jacobians for active joints
            # jacobians is (batch, num_links, full_dim, 3)
            
            # Create a new jacobian tensor for active joints
            jacobians_reduced = torch.zeros(self.batch_size, jacobians.shape[1], q.shape[1], 3, device=self.device)
            
            for i, idx in enumerate(self.active_indices):
                jacobians_reduced[:, :, i, :] = jacobians[:, :, idx, :]
                
            # Add contributions from mimic joints
            for mimic_idx, (parent_idx, multiplier, offset) in self.mimic_info.items():
                # Find which active index corresponds to parent_idx
                if parent_idx in self.active_indices:
                    active_pos = self.active_indices.index(parent_idx)
                    jacobians_reduced[:, :, active_pos, :] += jacobians[:, :, mimic_idx, :] * multiplier
            
            jacobians = jacobians_reduced

        # so we transpose the last two dimensions to get a
        # jacobian of shape (batch_size, num_links, 3, root_dim)
        # and then reshape it to (batch_size, num_links * 3, root_dim)
        jacobian = jacobians[:, self.link_indices, :, :].transpose(2,3).reshape(
                        self.batch_size, len(self.link_indices) * 3, q.shape[1])

        return (x, jacobian)




