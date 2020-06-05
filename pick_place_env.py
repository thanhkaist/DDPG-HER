import numpy as np
from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceEnv

class SawyerMultiGoalPickPlaceEnv(SawyerReachPushPickPlaceEnv):
    def __init__(self, random_init=True, task_type='pick_place'):
        super().__init__(random_init,task_type='pick_place')
        self.reward_type = 'sparse'
        self.distance_threshold = 0.07


    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, _, reachDist, _, pushDist, pickRew, _, placingDist = self.compute_reward(action, obs_dict)
        self.curr_path_length +=1

        goal_dist = placingDist if self.task_type == 'pick_place' else pushDist

        if self.task_type == 'reach':
            success = float(reachDist <= 0.05)
        else:
            success = float(goal_dist <= 0.07)

        info = {'reachDist': reachDist, 'pickRew':pickRew, 'epRew' : reward, 'goalDist': goal_dist, 'is_success': success}
        info['goal'] = self.goal
        if self.reward_type == 'sparse':
            reward = -(goal_dist > self.distance_threshold).astype(np.float32)
        return ob, reward, False, info


    def get_gym_observation_dict(self,obs_dict):
        return dict(
            observation = obs_dict['state_observation'],
            desired_goal = obs_dict['state_desired_goal'],
            achieved_goal = obs_dict['state_achieved_goal']
        )
    
    def compute_reward_a(self, achieved_goal, goal, info):
        
        d = np.linalg.norm(achieved_goal - goal,axis=-1)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
        return compute_reward(self, actions, obs)[0]

    def compute_reward(self, actions, obs):
        #obs = dict(
        #    state_observation = obs['observation'],
        #    state_desired_goal = obs['desired_goal'],
        #    state_achieved_goal = obs['achieved_goal']
        #)
        
        obs1 = obs['state_observation']
        objPos = obs1[3:6]
        fingerCOM  =  obs1[0:3]
        heightTarget = self.heightTarget
        goal = obs['state_desired_goal']

        def compute_reward_reach(actions, obs):
            del actions
            del obs

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachDist = np.linalg.norm(fingerCOM - goal)
            reachRew = c1*(self.maxReachDist - reachDist) + c1*(np.exp(-(reachDist**2)/c2) + np.exp(-(reachDist**2)/c3))
            reachRew = max(reachRew, 0)
            reward = reachRew
            return [reward, reachRew, reachDist, None, None, None, None, None]

        def compute_reward_push(actions, obs):
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            del actions
            del obs

            assert np.all(goal == self.get_site_pos('goal_push'))
            reachDist = np.linalg.norm(fingerCOM - objPos)
            pushDist = np.linalg.norm(objPos[:2] - goal[:2])
            reachRew = -reachDist
            if reachDist < 0.05:
                pushRew = 1000*(self.maxPushDist - pushDist) + c1*(np.exp(-(pushDist**2)/c2) + np.exp(-(pushDist**2)/c3))
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0
            reward = reachRew + pushRew
            return [reward, reachRew, reachDist, pushRew, pushDist, None, None, None]

        def compute_reward_pick_place(actions, obs):
            del obs

            reachDist = np.linalg.norm(objPos - fingerCOM)
            placingDist = np.linalg.norm(objPos - goal)
            assert np.all(goal == self.get_site_pos('goal_pick_place'))

            def reachReward():
                reachRew = -reachDist
                reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
                zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])

                if reachDistxy < 0.05:
                    reachRew = -reachDist
                else:
                    reachRew =  -reachDistxy - 2*zRew

                #incentive to close fingers when reachDist is small
                if reachDist < 0.05:
                    reachRew = -reachDist + max(actions[-1],0)/50

                return reachRew , reachDist

            def pickCompletionCriteria():
                tolerance = 0.01
                if objPos[2] >= (heightTarget- tolerance):
                    return True
                else:
                    return False

            if pickCompletionCriteria():
                self.pickCompleted = True


            def objDropped():
                return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)
                # Object on the ground, far away from the goal, and from the gripper
                # Can tweak the margin limits

            def orig_pickReward():
                hScale = 100
                if self.pickCompleted and not(objDropped()):
                    return hScale*heightTarget
                elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
                    return hScale* min(heightTarget, objPos[2])
                else:
                    return 0

            def placeReward():
                c1 = 1000
                c2 = 0.01
                c3 = 0.001
                cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
                if cond:
                    placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                    placeRew = max(placeRew,0)
                    return [placeRew , placingDist]
                else:
                    return [0 , placingDist]

            reachRew, reachDist = reachReward()
            pickRew = orig_pickReward()
            placeRew , placingDist = placeReward()
            assert ((placeRew >=0) and (pickRew>=0))
            reward = reachRew + pickRew + placeRew

            return [reward, reachRew, reachDist, None, None, pickRew, placeRew, placingDist]

        if self.task_type == 'reach':
            return compute_reward_reach(actions, obs)
        elif self.task_type == 'push':
            return compute_reward_push(actions, obs)
        elif self.task_type == 'pick_place':
            return compute_reward_pick_place(actions, obs)
        else:
            raise NotImplementedError
    
    
    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, objPos))
        targetPos = self._state_goal

        return dict(
            observation = flat_obs,
            desired_goal = targetPos,
            achieved_goal = objPos
        )