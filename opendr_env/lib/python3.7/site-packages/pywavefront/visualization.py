# ----------------------------------------------------------------------------
# PyWavefront
# Copyright (c) 2018 Kurt Yoder
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of PyWavefront nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------
import pyglet
from pyglet.gl import *

from pywavefront import Wavefront
from pywavefront.mesh import Mesh
from pywavefront.material import Material

# Mock the _nearest_pow2 method to allow for npot textures
def same(v):
    return v

pyglet.image._nearest_pow2 = same

VERTEX_FORMATS = {
    'V3F': GL_V3F,
    'C3F_V3F': GL_C3F_V3F,
    'N3F_V3F': GL_N3F_V3F,
    'T2F_V3F': GL_T2F_V3F,
    # 'C3F_N3F_V3F': GL_C3F_N3F_V3F,  # Unsupported
    'T2F_C3F_V3F': GL_T2F_C3F_V3F,
    'T2F_N3F_V3F': GL_T2F_N3F_V3F,
    # 'T2F_C3F_N3F_V3F': GL_T2F_C3F_N3F_V3F,  # Unsupported
}


def draw(instance, lighting_enabled=True, textures_enabled=True):
    """Generic draw function"""
    # Draw Wavefront instance
    if isinstance(instance, Wavefront):
        draw_materials(instance.materials, lighting_enabled=lighting_enabled, textures_enabled=textures_enabled)
    # Draw single material
    elif isinstance(instance, Material):
        draw_material(instance, lighting_enabled=lighting_enabled, textures_enabled=textures_enabled)
    # Draw dict of materials
    elif isinstance(instance, dict):
        draw_materials(instance, lighting_enabled=lighting_enabled, textures_enabled=textures_enabled)
    else:
        raise ValueError("Cannot figure out how to draw: {}".format(instance))


def draw_materials(materials, lighting_enabled=True, textures_enabled=True):
    """Draw a dict of meshes"""
    for name, material in materials.items():
        draw_material(material, lighting_enabled=lighting_enabled, textures_enabled=textures_enabled)


def draw_material(material, face=GL_FRONT_AND_BACK, lighting_enabled=True, textures_enabled=True):
    """Draw a single material"""
    if material.gl_floats is None:
        material.gl_floats = (GLfloat * len(material.vertices))(*material.vertices)
        material.triangle_count = len(material.vertices) / material.vertex_size

    vertex_format = VERTEX_FORMATS.get(material.vertex_format)
    if not vertex_format:
        raise ValueError("Vertex format {} not supported by pyglet".format(material.vertex_format))

    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glCullFace(GL_BACK)

    if textures_enabled:
        # Fall back to ambient texture if no diffuse
        texture = material.texture or material.texture_ambient
        if texture and material.has_uvs:
            bind_texture(texture)
        else:
            glDisable(GL_TEXTURE_2D)

    
    if lighting_enabled and material.has_normals:
        glMaterialfv(face, GL_DIFFUSE, gl_light(material.diffuse))
        glMaterialfv(face, GL_AMBIENT, gl_light(material.ambient))
        glMaterialfv(face, GL_SPECULAR, gl_light(material.specular))
        glMaterialfv(face, GL_EMISSION, gl_light(material.emissive))
        glMaterialf(face, GL_SHININESS, min(128.0, material.shininess))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
    else:
        glDisable(GL_LIGHTING)
        glColor4f(*material.ambient)

    glInterleavedArrays(vertex_format, 0, material.gl_floats)
    glDrawArrays(GL_TRIANGLES, 0, int(material.triangle_count))

    glPopAttrib()
    glPopClientAttrib()


def gl_light(lighting):
    """Return a GLfloat with length 4, containing the 4 lighting values."""
    return (GLfloat * 4)(*(lighting))


def bind_texture(texture):
    """Draw a single texture"""
    if not getattr(texture, 'image', None):
        texture.image = load_image(texture.find())

    glEnable(texture.image.target)
    glBindTexture(texture.image.target, texture.image.id)
    if texture.options.clamp == "on":
        glTexParameterf(texture.image.target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(texture.image.target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    else:
        glTexParameterf(texture.image.target, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(texture.image.target, GL_TEXTURE_WRAP_T, GL_REPEAT)


def load_image(name):
    """Load an image"""
    image = pyglet.image.load(name)
    texture = image.get_texture()
    # verify_dimensions(texture)
    return texture


def verify_dimensions(image):
    verify(image, 'width')
    verify(image, 'height')


def verify(image, dimension):
    value = image.__getattribute__(dimension)

    while value > 1:
        div_float = float(value) / 2.0
        div_int = int(div_float)

        if not (div_float == div_int):
            raise Exception('image %s is %d, which is not a power of 2' % (
                dimension, image.__getattribute__(dimension)))

        value = div_int
