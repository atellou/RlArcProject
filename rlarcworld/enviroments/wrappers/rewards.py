import numpy as np


def compute_pixel_correctnes(current_grid, target_grid):
    return np.sum(np.abs(current_grid - target_grid))



        if terminated:
            logger.info("Successfully completed environment.")
            # Maximum reward: expected to encourage exploration
            reward = abs(self._reward_storage)
        elif reward != 0 and submission:
            logger.warning("Attempting to submit with reward equal {}.".format(reward))
            # Maximum penalization
            reward = -1 * self.size * self.size * self.color_values