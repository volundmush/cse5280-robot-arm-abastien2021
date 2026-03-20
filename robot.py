import os
import urllib.request
import zipfile
from typing import List
import numpy as np
from numpy.linalg import norm
from vedo import Arrow, Sphere, LinearTransform  # type: ignore

def prepare_robot():
    if not os.path.exists("robot/"):
        url = "https://www.dropbox.com/scl/fi/uewvrcempf2wf2jp7bcb8/robot.zip?rlkey=7uwz1ne94hxyinub8x16y93em&dl=1"
        urllib.request.urlretrieve(url, "robot.zip")
        with zipfile.ZipFile("robot.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove("robot.zip")

class RobotArm:
    """
    Robot arm with:
      - forward kinematics
      - Jacobian-based IK
      - persistent vedo meshes updated in-place
    """

    def __init__(self, partLengths, arm_location, mesh_scale: float = 1.0):
        # Arm location (position of the first frame)
        self.arm_location = np.array(arm_location, dtype=float)

        # Lengths
        self.L1, self.L2, self.L3, self.L4 = partLengths

        # Mesh scale
        self.mesh_scale = mesh_scale

        # Store source meshes (do not mutate these directly)
        self.source_parts = self.load_robot_parts()

        # Constants
        self.delta_phi = 0.1              # finite-difference step, in degrees
        self.target = np.array([0, 100, 200], dtype=float)
        self.target_tolerance = 0.05
        self.target_lambda = 0.05
        self.convergence = 0.02
        self.iteration_limit = 1000

        # Live meshes for rendering
        self.meshes = None
        self.initialize_meshes()

    def initialize_meshes(self):
        """
        Create robot meshes once for this robot instance.
        These are the only meshes used during animation.
        """
        Base  = self.source_parts[0].clone()
        Part1 = self.source_parts[1].clone()
        Part2 = self.source_parts[2].clone()
        Part3 = self.source_parts[3].clone()
        Part4 = self.createCoordinateFrameMesh()

        self.meshes = [Base, Part1, Part2, Part3, Part4]

    def RotationMatrix(self, theta, axis_name):
        """
        Single-axis rotation matrix.
        theta is interpreted in degrees.
        """
        c = np.cos(theta * np.pi / 180.0)
        s = np.sin(theta * np.pi / 180.0)

        if axis_name == "x":
            rotation_matrix = np.array([
                [1, 0,  0],
                [0, c, -s],
                [0, s,  c]
            ])
        elif axis_name == "y":
            rotation_matrix = np.array([
                [ c, 0, s],
                [ 0, 1, 0],
                [-s, 0, c]
            ])
        elif axis_name == "z":
            rotation_matrix = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])
        else:
            raise ValueError(f"Unknown axis_name: {axis_name}")

        return rotation_matrix

    def createCoordinateFrameMesh(self):
        """
        Returns a mesh representing a coordinate frame.
        """
        shaft_radius = 0.5 * self.mesh_scale
        head_radius = 1.0 * self.mesh_scale
        alpha = 1.0
        unit = 120 * self.mesh_scale

        x_axisArrow = Arrow(
            start_pt=(0, 0, 0),
            end_pt=(unit, 0, 0),
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            res=12,
            c="red",
            alpha=alpha,
        )

        y_axisArrow = Arrow(
            start_pt=(0, 0, 0),
            end_pt=(0, unit, 0),
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            res=12,
            c="green",
            alpha=alpha,
        )

        z_axisArrow = Arrow(
            start_pt=(0, 0, 0),
            end_pt=(0, 0, unit),
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            res=12,
            c="blue",
            alpha=alpha,
        )

        originDot = Sphere(pos=[0, 0, 0], c="black", r=1.0 * self.mesh_scale)

        F = x_axisArrow + y_axisArrow + z_axisArrow + originDot
        return F

    def getLocalFrameMatrix(self, R_ij, t_ij):
        """
        Homogeneous transform matrix T_ij from rotation R_ij and translation t_ij.
        """
        T_ij = np.block([
            [R_ij, t_ij],
            [np.zeros((1, 3)), np.array([[1]])]
        ])
        return T_ij

    def forward_kinematics(self, Phi):
        """
        Returns:
            T_00, T_01, T_02, T_03, T_04, e
        where e is the end-effector position.
        """
        radius = 0.4 * self.mesh_scale

        phi1 = Phi[0]

        # Base
        R_00 = self.RotationMatrix(0, axis_name="z")
        t_00 = np.copy(self.arm_location)
        t_00[-1] = 0
        T_00 = self.getLocalFrameMatrix(R_00, t_00)

        # Frame 1
        R_01 = self.RotationMatrix(phi1, axis_name="z")
        t_01 = self.arm_location
        T_01 = self.getLocalFrameMatrix(R_01, t_01)

        # Frame 2
        phi2 = Phi[1]
        R_12 = self.RotationMatrix(phi2, axis_name="y")
        t_12 = np.array([[0.0], [0.0], [self.L1 + 2 * radius]])
        T_12 = self.getLocalFrameMatrix(R_12, t_12)
        T_02 = T_01 @ T_12

        # Frame 3
        phi3 = Phi[2]
        R_23 = self.RotationMatrix(phi3, axis_name="y")
        t_23 = np.array([[0.0], [0.0], [self.L2 + 2 * radius]])
        T_23 = self.getLocalFrameMatrix(R_23, t_23)
        T_03 = T_01 @ T_12 @ T_23

        # Frame 4 / end effector
        phi4 = Phi[3]
        R_34 = self.RotationMatrix(phi4, axis_name="y")
        t_34 = np.array([[-28.4 * self.mesh_scale], [0.0], [self.L3 + radius]])
        T_34 = self.getLocalFrameMatrix(R_34, t_34)
        T_04 = T_01 @ T_12 @ T_23 @ T_34

        e = T_04[0:3, -1]
        return T_00, T_01, T_02, T_03, T_04, e

    def get_pose_transforms(self, Phi):
        """
        Convenience function for rendering.
        """
        T_00, T_01, T_02, T_03, T_04, _ = self.forward_kinematics(Phi)
        return [T_00, T_01, T_02, T_03, T_04]

    def update_pose(self, Phi):
        """
        Update the robot's existing meshes in-place.
        No cloning, no mesh accumulation.
        """
        transforms = self.get_pose_transforms(Phi)

        # Re-clone from the original neutral meshes each frame to avoid
        # transform accumulation issues across frames.
        Base  = self.source_parts[0].clone()
        Part1 = self.source_parts[1].clone()
        Part2 = self.source_parts[2].clone()
        Part3 = self.source_parts[3].clone()
        Part4 = self.createCoordinateFrameMesh()

        new_meshes = [Base, Part1, Part2, Part3, Part4]

        for mesh, T in zip(new_meshes, transforms):
            mesh.apply_transform(LinearTransform(T))

        self.meshes = new_meshes
        return self.meshes



    def jacobian_matrix(self, phi):
        """
        Numerical Jacobian of end-effector position wrt joint angles.
        """
        # Keep everything in degrees to match the rest of your code
        step = self.delta_phi

        _, _, _, _, _, e = self.forward_kinematics(phi)

        _, _, _, _, _, e_phi1_delta = self.forward_kinematics(phi + np.array([step, 0, 0, 0]))
        e_phi1_derive = (e_phi1_delta - e) / step

        _, _, _, _, _, e_phi2_delta = self.forward_kinematics(phi + np.array([0, step, 0, 0]))
        e_phi2_derive = (e_phi2_delta - e) / step

        _, _, _, _, _, e_phi3_delta = self.forward_kinematics(phi + np.array([0, 0, step, 0]))
        e_phi3_derive = (e_phi3_delta - e) / step

        jacobian = np.concatenate(
            (
                e_phi1_derive.reshape((3, 1)),
                e_phi2_derive.reshape((3, 1)),
                e_phi3_derive.reshape((3, 1)),
            ),
            axis=1,
        )

        return jacobian

    def inverse_kinematics_newton(self, initial_phi, record_every=20, motion_threshold=10):
        """
        Runs IK and stores only joint angles.

        Returns
        -------
        trajectory : (N, 4) ndarray
            Recorded joint-angle states.
        """
        phi = np.array(initial_phi, dtype=float).copy()
        _, _, _, _, _, e = self.forward_kinematics(phi)

        target = np.array(self.target, dtype=float)
        target_lambda = self.target_lambda
        convergence = self.convergence
        target_tolerance = self.target_tolerance
        iteration_limit = self.iteration_limit

        recorder = [phi.copy()]
        iteration = 0
        e_accumulate = 0.0

        while np.linalg.norm(target - e) > target_tolerance:
            iteration += 1

            J = self.jacobian_matrix(phi)
            J_pinv = np.linalg.pinv(J)

            e_delta = target_lambda * (target - e)
            phi_delta = J_pinv @ e_delta
            phi_delta = np.append(phi_delta, [0.0])

            phi = phi + phi_delta

            e_previous = e
            _, _, _, _, _, e = self.forward_kinematics(phi)
            e_accumulate += np.linalg.norm(e_previous - e)

            if iteration % record_every == 0 or e_accumulate > motion_threshold:
                recorder.append(phi.copy())
                e_accumulate = 0.0

            if np.linalg.norm(e_previous - e) < convergence:
                break

            if iteration > iteration_limit:
                break

        # Always include final pose
        if not np.allclose(recorder[-1], phi):
            recorder.append(phi.copy())

        return np.array(recorder)


    def load_robot_parts(self) -> list:
        from vedo import load
        robot_dir = "robot/"
        Base = load(robot_dir + "Base.stl").color("purple5").scale(self.mesh_scale)
        BaseRot = load(robot_dir + "BaseRot.stl").color("purple4").scale(self.mesh_scale)
        Humerus = load(robot_dir + "Humerus.stl").color("gray5").scale(self.mesh_scale)
        Radius = load(robot_dir + "Radius.stl").color("red5").scale(self.mesh_scale)
        return [Base, BaseRot, Humerus, Radius]