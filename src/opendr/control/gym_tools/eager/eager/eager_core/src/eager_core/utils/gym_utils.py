import gym
import numpy as np
from gym.spaces.box import Box
from typing import Type
from eager_core.utils.file_utils import substitute_xml_args
from eager_core.srv import BoxFloat32Data, BoxFloat32DataResponse
from eager_core.srv import BoxUInt8Data, BoxUInt8DataResponse
from eager_core.msg import Space

def get_space_from_def(object: dict) -> gym.Space:
    if object['type'] == 'boxf32':
        if 'shape' in object:
            high = object['high']
            low = object['low']
            shape = object['shape']
            return Box(low, high, shape, dtype=np.float32)
        else:
            high = np.array(object['high'], dtype=np.float32)
            low = np.array(object['low'], dtype=np.float32)
            return Box(low, high, dtype=np.float32)
    elif object['type'] == 'boxu8':
        if 'shape' in object:
            high = object['high']
            low = object['low']
            shape = object['shape']
            return Box(low, high, shape, dtype=np.uint8)
        else:
            high = np.array(object['high'], dtype=np.uint8)
            low = np.array(object['low'], dtype=np.uint8)
            return Box(low, high, dtype=np.uint8)
    else:
        raise NotImplementedError('Unknown space type:', object['type'])
        
def get_def_from_space(space: gym.Space) -> dict:
    if type(space) is Box:
        object = dict()
        object['high'] = space.high
        object['low'] = space.low
        if space.dtype is np.dtype(np.float32):
            object['type'] = 'boxf32'
            return object
        elif space.dtype is np.dtype(np.uint8):
            object['type'] = 'boxu8'
            return object
        else:
            raise NotImplementedError('Unknown space type:', space.dtype)
    else:
        raise NotImplementedError('Unknown space type:', type(space))
        
def get_message_from_space(space: gym.Space) -> Type:
    if type(space) is Box:
        if space.dtype is np.dtype(np.float32):
            return BoxFloat32Data
        elif space.dtype is np.dtype(np.uint8):
            return BoxUInt8Data
        else:
            raise NotImplementedError('Unknown space type:', space.dtype)
    else:
        raise NotImplementedError('Unknown space type:', type(space))

def get_response_from_space(space: gym.Space) -> Type:
    if type(space) is Box:
        if space.dtype is np.dtype(np.float32):
            return BoxFloat32DataResponse
        elif space.dtype is np.dtype(np.uint8):
            return BoxUInt8DataResponse
        else:
            raise NotImplementedError('Unknown space type:', space.dtype)
    else:
        raise NotImplementedError('Unknown space type:', type(space))
        
def get_def_from_space_msg(msg: Space) -> dict:
    object = dict()
    object['type'] = msg.type
    object['high'] = [] 
    for value in msg.high:
        object['high'].append(substitute_xml_args(value))
    object['low'] = []
    for value in msg.low:
        object['low'].append(substitute_xml_args(value))
    return object
    
def get_space_msg_from_def(object: dict) -> Space:
    msg = Space()
    msg.type = object['type']
    msg.high = []
    for value in object['high']:
    	msg.high.append(str(value))
    msg.low = []
    for value in object['low']:
    	msg.low.append(str(value))
    return msg
    
def get_space_from_space_msg(msg: Space) -> gym.Space:
		space_def = get_def_from_space_msg(msg)
		return get_space_from_def(space_def)
		
def get_space_msg_from_space(space: gym.Space) -> Space:
		space_def = get_def_from_space(space)
		return get_space_msg_from_def(space_def)
