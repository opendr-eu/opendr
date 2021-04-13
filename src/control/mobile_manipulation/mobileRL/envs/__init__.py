from .tasks import RestrictedWsTask, RndStartRndGoalsTask, SimpleObstacleTask, HouseExpoTask, AdversarialTask, \
    HierarchiecalAdversarial
from .tasks_chained import PickNPlaceChainedTask, DoorChainedTask, DrawerChainedTask

_all_tasks = [RndStartRndGoalsTask, RestrictedWsTask, SimpleObstacleTask, AdversarialTask, HouseExpoTask,
              PickNPlaceChainedTask, DoorChainedTask, DrawerChainedTask, HierarchiecalAdversarial]
ALL_TASKS = {task.taskname().lower(): task for task in _all_tasks}
