# Copyright 2020-2024 OpenDR European Project
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

from controller import Supervisor
import math
import random

robot = Supervisor()
root_node = robot.getRoot()
root_children_field = root_node.getField('children')

# add plants
new_field_node_string = ''
col = 0
row = 0
for i in range(0, 40):
    row = 0
    for j in range(0, 14):
        angle = random.uniform(0, 1) * math.pi
        new_field_node_string += \
            f" SimpleTree {{ translation {col} {row} 0 rotation 0 0 1 {angle} name \"plant_{i}_{j}\" \
            type \"hazel tree\" enableBoundingObject FALSE }}"
        row += 8
    col += 5

size_x = col * 0.2
size_y = row * 0.2
tx = -size_x
ty = -size_y * 0.5
new_field_node_string = \
    f"DEF FIELD_SOIL Transform {{ translation {tx} {ty} -0.08 scale 0.2 0.2 0.1 children [ {new_field_node_string} ] }}"
root_children_field.importMFNodeFromString(-1, new_field_node_string)

# add field mud
ground_size_x = size_x + 10
ground_size_y = size_y + 10
ground_tx = -size_x / 2
root_children_field.importMFNodeFromString(-1, f"DEF FIELD_MUD SolidBox {{ \
   translation {ground_tx} 0 0 name \"field mud\" size {ground_size_x} {ground_size_y} 0.1 \
   contactMaterial \"ground\"appearance Soil {{ textureTransform TextureTransform {{ scale 100 100 }} color \
   0.233341 0.176318 0.112779 }} }}")

# add obstacles
obstacles = ['CharacterSkin', 'Cat', 'Cow', 'Deer', 'Dog', 'Fox', 'Horse', 'Rabbit', 'Sheep']
for i in range(0, 3):
    valid = bool(random.getrandbits(1))
    print(valid)
    if valid:
        index = random.randrange(len(obstacles))
        r = []
        j = 0
        for j in range(0, 3):
            r.append(random.uniform(0, 1))
            print(r[j])
        pos_x = random.uniform(0, 1) * size_x + tx
        pos_y = random.uniform(0, 1) * size_y + ty
        angle = random.uniform(0, 1) * 2 * math.pi
        if index == 0:
            node_name = f"human_{j}"
            model_names = ["Sandra", "Robert", "Anthony", "Sophia"]
            human_index = random.randrange(len(model_names))
            node_string = \
                f"Robot {{ translation {pos_x} {pos_y} 0.05 rotation 0 0 1 {angle} children [ \
                  CharacterSkin {{ model \"{model_names[human_index]}\" }}\
                ] name \"{node_name}\" }}"
        else:
            node_string = f"{obstacles[index]} {{ translation {pos_x} {pos_y} 0.05 rotation 0 0 1 {angle} }}"
        root_children_field.importMFNodeFromString(-1, node_string)
