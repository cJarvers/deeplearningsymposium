from roboschool.scene_abstract import SingleRobotEmptyScene
from roboschool.gym_mujoco_xml_env import *
import numpy as np


class ParametrizedDoublePendulum(RoboschoolMujocoXmlEnv):
    '''
    Two-link continuous control version of classic cartpole problem.
    Keep two-link pendulum upright by moving the 1-D cart.
    Similar to MuJoCo InvertedDoublePendulum task.
    '''
    def __init__(self, meta=False):
        RoboschoolMujocoXmlEnv.__init__(self, 'inverted_double_pendulum.xml', 'cart', action_dim=1, obs_dim=9)
        self.alive_bonus = 10 # np.random.uniform(low=8.0, high=12.0)

        if not meta:
            self.dist_penalty = np.random.uniform(low=1e-3, high=1e-1)
            self.start_range = np.random.uniform(low=-0.15, high=0.15, size=(2,)) #
            self.motor_torque = np.random.uniform(low=0, high=5, size=(2,)) #
            self.inverted_acts = np.random.uniform(low=0, high=1)
            self.actuation = np.random.randint(low=185, high=215)
            self.goal_pos = np.random.uniform(low=0.2, high=0.4)
            self.gravity = np.random.uniform(low=0.01, high=4.9)
        else:
            self.start_range = np.random.uniform(low=-0.25, high=0.25, size=(2,))
            self.motor_torque = np.random.uniform(low=0, high=3, size=(2,))
            self.inverted_acts = np.random.uniform(low=0, high=1.0)
            self.actuation = np.random.randint(low=175, high=225)
            self.goal_pos = np.random.uniform(low=0.3, high=0.4)
            self.gravity = np.random.uniform(low=4.9, high=9.8)
            self.dist_penalty = np.random.uniform(low=1e-3, high=1e-2)
        self.inv_switch = 1 if self.inverted_acts >= 0.5 else 0

    def get_params(self):
        return [self.dist_penalty, self.start_range[0], self.start_range[1], self.motor_torque[0],
                self.motor_torque[1], self.inverted_acts, self.actuation, self.goal_pos, self.gravity]

    def create_single_player_scene(self):
        return SingleRobotEmptyScene(gravity=self.gravity, timestep=0.0165, frame_skip=1)

    def robot_specific_reset(self):
        self.pole2 = self.parts["pole2"]
        self.slider = self.jdict["slider"]
        self.j1 = self.jdict["hinge"]
        self.j2 = self.jdict["hinge2"]
        u = self.np_random.uniform(low=self.start_range[0], high=self.start_range[1], size=[2])
        self.j1.reset_current_position(float(u[0]), 0)
        self.j2.reset_current_position(float(u[1]), 0)
        self.j1.set_motor_torque(self.motor_torque[0])
        self.j2.set_motor_torque(self.motor_torque[1])

    def apply_action(self, a):
        if self.inv_switch:
            a = a * -1
        assert( np.isfinite(a).all() )
        self.slider.set_motor_torque( self.actuation*float(np.clip(a[0], -1, +1)) )

    def calc_state(self):
        theta, theta_dot = self.j1.current_position()
        gamma, gamma_dot = self.j2.current_position()
        self.theta_dot = theta_dot
        self.gamma_dot = gamma_dot
        x, vx = self.slider.current_position()
        self.vx = vx
        self.pos_x, _, self.pos_y = self.pole2.pose().xyz()
        assert( np.isfinite(x) )
        return np.array([
            x, vx,
            self.pos_x,
            np.cos(theta), np.sin(theta), theta_dot,
            np.cos(gamma), np.sin(gamma), gamma_dot,
            ])

    def step(self, a):
        self.apply_action(a)
        self.scene.global_step()
        state = self.calc_state()  # sets self.pos_x self.pos_y
        # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
        # using <site> tag in original xml, upright position is 0.6 + 0.6 = 1.2, difference +0.3
        dist_penalty = self.dist_penalty * self.pos_x ** 2 + (self.pos_y + self.goal_pos - 2) ** 2
        vel_penalty = 0 #self.vel_penatly * (self.theta_dot + self.gamma_dot + self.vx)
        alive_bonus = self.alive_bonus
        done = self.pos_y + self.goal_pos <= 1
        self.rewards = [float(alive_bonus), float(-dist_penalty), float(-vel_penalty)]
        self.frame  += 1
        self.done   += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)
        return state, sum(self.rewards), done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0,1.2,1.2, 0,0,0.5)
