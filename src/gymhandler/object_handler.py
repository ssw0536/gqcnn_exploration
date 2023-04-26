from isaacgym import gymapi, gymutil
from isaacgym import gymtorch


class GymObjectHandler(object):
    def __init__(self, gym):
        self._gym = gym

    def reset(self):
        pass

    def get_current_pose(self):
        pass

    def add_to_env(self):
        pass
