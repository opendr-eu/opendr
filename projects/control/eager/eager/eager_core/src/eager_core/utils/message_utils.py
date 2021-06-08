from eager_core.srv import BoxFloat32Data, BoxFloat32DataResponse
from eager_core.srv import BoxUInt8Data, BoxUInt8DataResponse

def get_message_from_def(object):
    if object['type'] == 'boxf32':
        return BoxFloat32Data
    elif object['type'] == 'boxu8':
        return BoxUInt8Data
    else:
        raise NotImplementedError('Unknown space type:', object['type'])

def get_response_from_def(object):
    if object['type'] == 'boxf32':
        return BoxFloat32DataResponse
    elif object['type'] == 'boxu8':
        return BoxUInt8DataResponse
    else:
        raise NotImplementedError('Unknown space type:', object['type'])

def get_value_from_def(object):
    if object['type'] == 'boxf32':
        return 0.0
    elif object['type'] == 'boxu8':
        return 0
    else:
        raise NotImplementedError('Unknown message type:', object['type'])

def get_length_from_def(object):
    if 'shape' in object and object['shape']:
        length = 1
        for element in object['shape']:
            length *= element
    else:
        length = len(object['high'])
    return length

def get_dtype_from_def(object):
    if 'f32' in object['type'] == 'boxf32':
        return 'float32'
    elif 'u8' in object['type']:
        return 'uint8'
    else:
        raise NotImplementedError('Unknown space type:', object['type'])