from .tasks import RestrictedWsTask, RndStartRndGoalsTask
from .tasks_chained import PickNPlaceChainedTask, DoorChainedTask, DrawerChainedTask

_all_tasks = [RndStartRndGoalsTask, RestrictedWsTask, PickNPlaceChainedTask, DoorChainedTask, DrawerChainedTask]
ALL_TASKS = {task.taskname().lower(): task for task in _all_tasks}
