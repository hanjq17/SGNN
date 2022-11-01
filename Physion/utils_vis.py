import numpy as np
def create_instance_colors(n):
    # TODO: come up with a better way to initialize instance colors
    return np.array([
        [1., 1., 0., 1.],
        #[1., 0., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
        [1., 0., 1., 1.],
        [0., 1., 1., 1.],
        [1., 0., 0., 1.],
        [1., 1., 1., 1.],
        [1., 0.5, 0.5, 1.],
        [0.5, 0.5, 1., 1.],
        [0.5, 1., 0.5, 1.],
        [1., 0.25, 0.25, 1.],
        [0.25, 1., 0.25, 1.],
        [0.25, 0.25, 1., 1.],
        [0.25, 0.1, 1., 1.],
        [0.25, 0.1, 0.1, 1.],
        [0.1, 0.1, 1., 1.],])[:n]


def convert_groups_to_colors(group, instance_colors, env=None):
    """
    Convert grouping to RGB colors of shape (n_particles, 4)
    :param grouping: [p_rigid, p_instance, physics_param]
    :return: RGB values that can be set as color densities
    group: [0, 1024, 1032]

    """
    # p_rigid: n_instance
    # p_instance: n_p x n_instance
    n_instance = len(group) - 1
    n_particles = group[-1]

    #p_rigid, p_instance = group[:2]
    #p = p_instance

    colors = np.empty((n_particles, 4))

    for instance_id in range(n_instance):
        st, end = group[instance_id], group[instance_id+1]
        colors[st:end] = instance_colors[instance_id]

    # print("colors", colors)
    return colors