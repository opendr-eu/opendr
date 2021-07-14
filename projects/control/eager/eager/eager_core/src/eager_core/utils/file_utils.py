import rospkg, rosparam
from roslaunch.substitution_args import resolve_args
from future.utils import raise_from


def load_yaml(package_name, object_name):
    try:
        pp = rospkg.RosPack().get_path(package_name)
        filename = pp + "/config/" + object_name + ".yaml"
        params = rosparam.load_file(filename)[0][0]
    except Exception as ex:
        raise_from(RuntimeError(('Unable to load %s from package %s' % object_name, package_name)), ex)
    return params


def substitute_xml_args(param):
    # substitute string
    if isinstance(param, str):
        param = resolve_args(param)
        return param

    # For every key in the dictionary (not performing deepcopy!)
    if isinstance(param, dict):
        for key in param:
            # If the value is of type `(Ordered)dict`, then recurse with the value
            if isinstance(param[key], dict):
                substitute_xml_args(param[key])
            # Otherwise, add the element to the result
            elif isinstance(param[key], str):
                param[key] = resolve_args(param[key])
