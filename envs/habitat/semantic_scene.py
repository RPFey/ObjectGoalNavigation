from distutils.log import info
from unicodedata import category
import gym
import numpy as np
import math
import habitat
import os

import cv2
import numpy as np
import skimage.morphology
from PIL import Image

from envs.utils.fmm_planner import FMMPlanner
import envs.utils.pose as pu
import agents.utils.visualization as vu
from constants import color_palette, coco_categories

class HM3D_Env(habitat.RLEnv):
    def __init__(self, args, rank, config_env, dataset):
        """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank
        super().__init__(config_env, dataset)

        # Specifying action and observation space
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                 args.frame_width),
                                                dtype='uint8')

        self.info = {}
        self.selem = skimage.morphology.disk(3)

        self.episode_no = 0
        self.stopped = False
        self.path_length = 0.
        self.last_sim_location = [0., 0., 0.]
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None
    
    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs.shape), 0., False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0

        action = self._plan(planner_inputs)

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)

        if action >= 0:

            # act
            action = {'action': action}
            self.last_action = action['action']
            obs, rew, done, info = self.step(action)

            # preprocess obs
            obs = self._preprocess_obs(obs)
            self.obs = obs
            self.info = info

            info['g_reward'] += rew
            return obs, rew, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs_shape), 0., False, self.info
    
    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        if args.visualize or args.print_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0 / args.map_resolution - gx1),
                          int(c * 100.0 / args.map_resolution - gy1)]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)

        # Deterministic Local Policy
        if stop and planner_inputs['found_goal'] == 1:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def _visualize(self, inputs):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            cv2.imwrite(fn, self.vis_image)

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args

        obs = super().reset()
        obs = self._preprocess_obs(obs)
        
        # Reset Attributes
        self.stopped = False
        self.path_length = 0.
        self.prev_distance = self._env.task.measurements.get_metrics()['distance_to_goal']
        self.starting_distance = self.prev_distance

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.goal_name = self._env.current_episode.object_category
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.col_width = 1
        self.count_forward_actions = 0
        self.last_action = None
        self.episode_no = self._env.current_episode.episode_id

        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        # Update State Information
        self.info.update(
            {
                "time": 0,
                'sensor_pose': [0., 0., 0.],
                'goal_cat_id': coco_categories[self._env.current_episode.object_category],
                'goal_name': self.goal_name,
                'g_reward': 0.
            }
        )

        agent_state = self._env.sim.get_agent_state()
        self.last_sim_location = [agent_state.position[2], agent_state.position[0], 
                         2 * math.atan2(agent_state.rotation.components[2], agent_state.rotation.components[0])]

        return obs, self.info

    def _preprocess_obs(self, obs):
        rgb = obs['rgb']
        self.rgb_vis = rgb[:, :, ::-1]

        depth = obs['depth']
        depth = self._preprocess_depth(depth, self.args.min_depth, self.args.max_depth)

        semantic = obs['semantic'][:, :, 0]
        semantic_map = np.zeros((rgb.shape[0], rgb.shape[1], self.args.num_sem_categories), dtype=np.bool)
        objects_id = np.unique(semantic)
        for id in objects_id:
            category_name = self.habitat_env.sim.semantic_scene.objects[id].category.name()
            if category_name in coco_categories.keys():
                semantic_map[:, :, coco_categories[category_name]] = np.bitwise_or(
                    semantic_map[:, :, coco_categories[category_name]], semantic == id
                )
        
        ds = self.args.env_frame_width // self.args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = rgb[ds // 2::ds, ds // 2::ds]
            depth = depth[ds // 2::ds, ds // 2::ds]
            semantic_map = semantic_map[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, semantic_map),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        action = action["action"]
        if action == 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            action = 3

        obs, rew, done, _ = super().step(action)

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

        spl, success, dist = 0., 0., 0.
        if done:
            spl, success, dist = self.get_metrics()
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success

        self.info['time'] += 1
        return obs, rew, done, self.info

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        agent_state = self._env.sim.get_agent_state()
        curr_sim_pose = [agent_state.position[2], agent_state.position[0], 
                         2 * math.atan2(agent_state.rotation.components[2], agent_state.rotation.components[0])]
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)

        dx = -1 * dx
        dy = -1 * dy
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        dist = self._env.task.measurements.get_metrics()['distance_to_goal']
        if dist <= 1.0:
            success = 1
        else:
            success = 0
        spl = min(success * self.starting_distance / self.path_length, 1)
        return spl, success, dist

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)

    def get_reward(self, observations):
        """ Calculate rewards """
        self.curr_distance = self._env.task.measurements.get_metrics()['distance_to_goal']
        reward = (self.prev_distance - self.curr_distance) * \
            self.args.reward_coeff

        self.prev_distance = self.curr_distance
        return reward

    def get_done(self, observations):
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        if self.info['time'] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            scene = filename[: -len(scene_dataset_ext)]
            scenes.append(scene)
    scenes.sort()
    return scenes

if __name__ == '__main__':
    def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
        points_topdown = []
        bounds = pathfinder.get_bounds()
        for point in points:
            # convert 3D x,z to topdown x,y
            px = (point[0] - bounds[0][0]) / meters_per_pixel
            py = (point[2] - bounds[0][2]) / meters_per_pixel
            points_topdown.append(np.array([px, py]))
        return points_topdown

    from habitat import make_dataset
    from habitat.config.default import get_config as cfg_env
    from arguments import get_args
    args = get_args()

    env_config = cfg_env("envs/habitat/configs/tasks/objectnav_hm3d.yaml")
    scenes = _get_scenes_from_folder("data/datasets/objectnav/hm3d/v1/train/content")

    env_config.defrost()
    env_config.SIMULATOR.SCENE_DATASET = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    env_config.DATASET.CONTENT_SCENES = scenes[:3]
    env_config.freeze()

    dataset = make_dataset(env_config.DATASET.TYPE, config=env_config.DATASET)
    env_config.defrost()
    env_config.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    env_config.freeze()

    env = HM3D_Env(args, 0, env_config, dataset)
    obs = env.reset()
    obs = env.step(1)
    obs = env.step(1)
    obs = env.step(0)
    obs = env.step(1)


