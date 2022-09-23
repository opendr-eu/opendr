// Auto-generated. Do not edit!

// (in-package ros_bridge.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class OpenDRPose2DKeypoint {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.kpt_name = null;
      this.x = null;
      this.y = null;
    }
    else {
      if (initObj.hasOwnProperty('kpt_name')) {
        this.kpt_name = initObj.kpt_name
      }
      else {
        this.kpt_name = '';
      }
      if (initObj.hasOwnProperty('x')) {
        this.x = initObj.x
      }
      else {
        this.x = 0;
      }
      if (initObj.hasOwnProperty('y')) {
        this.y = initObj.y
      }
      else {
        this.y = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type OpenDRPose2DKeypoint
    // Serialize message field [kpt_name]
    bufferOffset = _serializer.string(obj.kpt_name, buffer, bufferOffset);
    // Serialize message field [x]
    bufferOffset = _serializer.int32(obj.x, buffer, bufferOffset);
    // Serialize message field [y]
    bufferOffset = _serializer.int32(obj.y, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type OpenDRPose2DKeypoint
    let len;
    let data = new OpenDRPose2DKeypoint(null);
    // Deserialize message field [kpt_name]
    data.kpt_name = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [x]
    data.x = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [y]
    data.y = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.kpt_name);
    return length + 12;
  }

  static datatype() {
    // Returns string type for a message object
    return 'ros_bridge/OpenDRPose2DKeypoint';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '6ce75e74f73663ed82a3235764bc7edf';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Copyright 2020-2022 OpenDR European Project
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    
    # This message contains all relevant information for an OpenDR human pose 2D keypoint
    
    # The kpt_name according to https://github.com/opendr-eu/opendr/blob/master/docs/reference/lightweight-open-pose.md#notes
    string kpt_name
    
    # x and y pixel position on the input image, (0, 0) is top-left corner of image
    int32 x
    int32 y
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new OpenDRPose2DKeypoint(null);
    if (msg.kpt_name !== undefined) {
      resolved.kpt_name = msg.kpt_name;
    }
    else {
      resolved.kpt_name = ''
    }

    if (msg.x !== undefined) {
      resolved.x = msg.x;
    }
    else {
      resolved.x = 0
    }

    if (msg.y !== undefined) {
      resolved.y = msg.y;
    }
    else {
      resolved.y = 0
    }

    return resolved;
    }
};

module.exports = OpenDRPose2DKeypoint;
