# 统一约定
##############################################################################
# 世界系/标准飞镖机体系 b：
#   x_b: Right, y_b: Forward, z_b: Up
#   四元数 q / 旋转矩阵 R 表示机体系到世界系的旋转，即：v_world = R @ v_body
#   因此速度从世界系转换到机体系时使用：v_body = R.T @ v_world
#
# 空气动力学坐标系 a：保持原有约定不变（LUF）
#   x_a: Left, y_a: Up, z_a: Forward
#   因此 a 系到 b 系（FLU）的分量变换为：
#     x_b = -x_a
#     y_b = z_a
#     z_b = y_a
#   对应变换矩阵：
#     T_b_from_a = 
#
# 气动力神经网络输出接口约定：
#   out_model = [Mpitch_a, Mroll_a, Fx_a, Fy_a, Myaw_a, Fz_a]
#   其中力分量组装为 F_a = [Fx_a, Fy_a, Fz_a]
#   力矩分量重排为 M_a = [Mroll_a, Mpitch_a, Myaw_a]
################################################################################

import numpy as np

def quat_to_rot(q):
    # 机体系的坐标规范与世界系一致
    w,x,y,z = q
    norm = np.sqrt(w*w+x*x+y*y+z*z)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    w,x,y,z = w/norm,x/norm,y/norm,z/norm
    R_WB = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])
    return R_WB

def quat_derivative(q, omega):
    """
    根据机体系角速度计算四元数导数。
    约定：
    - q = [w, x, y, z]
    - q 表示 body -> world 的旋转
    - omega = [wx, wy, wz]，在 body frame 下表达
    数学形式：
        q_dot = 0.5 * q ⊗ [0, omega]
    参数
    ----
    q : array-like, shape (4,)
        当前四元数 [w, x, y, z]
    omega : array-like, shape (3,)
        当前角速度 [wx, wy, wz]，机体系下表达
    返回
    ----
    q_dot : np.ndarray, shape (4,)
        四元数导数 [w_dot, x_dot, y_dot, z_dot]
    """
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    w, x, y, z = q
    wx, wy, wz = omega
    q_dot = 0.5 * np.array([
        -x * wx - y * wy - z * wz,
         w * wx + y * wz - z * wy,
         w * wy + z * wx - x * wz,
         w * wz + x * wy - y * wx
    ])
    return q_dot

def normalize_quat(q, eps=1e-12):
    q = np.asarray(q, dtype=float)

    norm = np.linalg.norm(q)

    if norm < eps:
        raise ValueError("Quaternion norm is too small, cannot normalize.")

    return q / norm


def quat_to_euler(q):
    """
    将四元数转换为自定义欧拉角定义。

    约定：
    - q = [w, x, y, z]
    - q 表示 body -> world 的旋转
    - pitch: 绕 X 轴，右手方向为正
    - roll:  绕 Y 轴，右手方向为正
    - yaw:   绕 Z 轴，右手方向为正

    返回
    ----
    euler : np.ndarray, shape (3,)
        [pitch_x, roll_y, yaw_z]，单位 rad
    """
    from scipy.spatial.transform import Rotation as SciRot

    w, x, y, z = normalize_quat(q)
    rot = SciRot.from_quat([x, y, z, w])
    pitch_x, roll_y, yaw_z = rot.as_euler("xyz", degrees=False)
    return np.array([pitch_x, roll_y, yaw_z], dtype=np.float32)


class DartDynamics:

    def __init__(self, aero_model, config=None):

        self.aero = aero_model
        self.config = config or {}

        dynamics_cfg = self.config.get("dynamics", {})
        initial_state_cfg = self.config.get("initial_state", {})

        self.m = float(dynamics_cfg.get("mass", 0.2))

        self.g = float(dynamics_cfg.get("gravity", 9.81))

        self.V_ref = float(dynamics_cfg.get("v_ref", 18.0))

        inertia_diag = np.asarray(
            dynamics_cfg.get("inertia_diag", [0.02, 0.02, 0.02]),
            dtype=np.float32,
        )
        self.I = np.diag(inertia_diag)

        self.I_inv = np.linalg.inv(self.I)

        #设置飞镖的物理参数：质量、重力加速度、惯性矩阵、阻力系数

        self.pos = np.asarray(
            initial_state_cfg.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32
        )

        self.vel = np.asarray(
            initial_state_cfg.get("vel", [10.0, 0.0, 10.0]), dtype=np.float32
        )

        self.omega = np.asarray(
            initial_state_cfg.get("omega", [0.0, 0.0, 0.0]), dtype=np.float32
        )

        initial_quat = initial_state_cfg.get("quat")
        if initial_quat is not None:
            self.q = normalize_quat(
                np.asarray(initial_quat, dtype=np.float32)
            )
        else:
            # 当前坐标系约定：x 右、y 前、z 上
            # 这里初始化时让机体系 Y 轴严格对齐初始速度方向；
            # 同时用世界系上方向消除绕 Y 轴的旋转自由度。
            from scipy.spatial.transform import Rotation as SciRot

            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            world_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            speed = float(np.linalg.norm(self.vel))
            if speed < 1e-8:
                self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                y_axis = self.vel / speed
                y_axis = y_axis / np.linalg.norm(y_axis)

                x_axis = np.cross(y_axis, world_up)
                x_norm = float(np.linalg.norm(x_axis))
                if x_norm < 1e-8:
                    x_axis = np.cross(y_axis, world_right)
                    x_norm = float(np.linalg.norm(x_axis))
                x_axis = x_axis / x_norm

                z_axis = np.cross(x_axis, y_axis)
                z_axis = z_axis / np.linalg.norm(z_axis)

                R_bw = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
                quat_xyzw = SciRot.from_matrix(R_bw).as_quat().astype(np.float32)
                self.q = normalize_quat(np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)).astype(np.float32)
                # 根据初始速度计算初始姿态，使飞镖前轴与速度方向对齐，并固定绕 Y 轴的角度


    def step(self, delta, dt):

        R = quat_to_rot(self.q)

        v_body = R.T @ self.vel

        #机体系的坐标规范与世界系一致
        v_forward, v_right, v_up = v_body[1] , v_body[0] , v_body[2]

        V = np.linalg.norm(v_body) + 1e-6

        # 侧滑角 beta：左负右正；攻角 alpha：上负下正
        beta = np.arctan2(v_right, v_forward)
        alpha = -np.arctan2(v_up, v_forward)

        tmp = delta[2]
        delta[2] = delta[3]
        delta[3] = tmp

        out_model = self.aero.predict(alpha, beta, delta)
        out_model = np.asarray(out_model, dtype=np.float32)

        # 模型输出接口按照空气动力学坐标系 a(LUF) 定义：
        # [Mpitch_a, Mroll_a, Fx_a, Fy_a, Myaw_a, Fz_a]
        # 其中：Mpitch_a 绕 pitch 轴，Mroll_a 绕 roll 轴，Myaw_a 绕 yaw 轴。
        # 所以力矩分量重排为 [Mroll_a, Mpitch_a, Myaw_a] 再进入 b 系变换。
        F_a = np.array([out_model[2], out_model[3], out_model[5]], dtype=np.float32)
        M_a = np.array([out_model[0], out_model[4], out_model[1]], dtype=np.float32)

        T_b_from_a = np.array([
            [-1, 0, 0],  # x_b <- -x_a
            [0, 0, 1],  # y_b <- z_a
            [0, 1, 0],  # z_b <- y_a
        ], dtype=np.float32)

        F_body = T_b_from_a @ F_a
        M_body = T_b_from_a @ M_a

        F_body = (V / self.V_ref) ** 2 * F_body
        M_body = (V / self.V_ref) ** 2 * M_body

        F_world = R @ F_body

        F_gravity = np.array([0.0, 0.0, -self.m * self.g])

        F_total = F_world + F_gravity

        acc = F_total / self.m

        self.vel += acc * dt

        self.pos += self.vel * dt

        #采取半隐式欧拉方法，先更新速度，再更新位置，数值稳定性更高

        omega_dot = self.I_inv @ (
            M_body - np.cross(self.omega, self.I @ self.omega)
        )

        self.omega += omega_dot * dt

        q_dot = quat_derivative(self.q, self.omega)

        self.q += q_dot * dt
        self.q = normalize_quat(self.q)



        # 返回与最终四元数一致的旋转矩阵，避免奖励计算使用一步滞后的姿态
        R_final = quat_to_rot(self.q)
        euler_final = quat_to_euler(self.q)

        return {
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "quat": self.q.copy(),
            "euler": euler_final.copy(),
            "omega": self.omega.copy(),
            "force": F_total.copy(),
            "R": R_final.copy()
        }

