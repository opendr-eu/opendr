# ----------------------------------------------------------------------------
# PyWavefront
# Copyright (c) 2013 Kurt Yoder
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
# See: http://paulbourke.net/dataformats/mtl/

import pywavefront
from pathlib import Path, PureWindowsPath
import re

class TextureOptions:

    def __init__(self):
        """Set up options with sane defaults"""
        self.name = "default"
        self.blendu = "on"
        self.blendv = "on"
        self.bm = 1.0
        self.boost = 0.0
        self.cc = "off"
        self.clamp = "off"
        self.imfchan = "l"
        self.mm = (0.0, 1.0)
        self.o = (0.0, 0.0, 0.0)
        self.s = (1.0, 1.0, 1.0)
        self.t = (0.0, 0.0, 0.0)
        self.texres = None


class TextureOptionsParser:

    def __init__(self, line):
        self._line = line
        self._gen = None
        self._options = TextureOptions()
        self._dispatch = {
            '-blendu': self.parse_blendu,
            '-blendv': self.parse_blendv,
            '-bm': self.parse_bm,
            '-boost': self.parse_boost,
            '-cc': self.parse_cc,
            '-clamp': self.parse_clamp,
            '-imfchan': self.parse_imfchan,
            '-mm': self.parse_mm,
            '-o': self.parse_o,
            '-s': self.parse_s,
            '-t': self.parse_t,
            '-texres': self.parse_texres,
        }

    def parse(self):
        def create_generator():
            for t in self._line.split():
                yield t

        self._gen = create_generator()

        try:
            while True:
                item = next(self._gen)
                func = self._dispatch.get(item, None)
                if func:
                    func()
                else:
                    self._options.name = ' '.join([item] + list(self._gen))
        except StopIteration:
            pass

        return self._options

    def parse_blendu(self):
        """The -blendu option turns texture blending in the horizontal direction 
        (u direction) on or off.  The default is on.
        """
        self._options.blendu = next(self._gen)

    def parse_blendv(self):
        """The -blendv option turns texture blending in the vertical direction (v 
        direction) on or off.  The default is on.
        """
        self._options.blendv = next(self._gen)

    def parse_bm(self):
        """The -bm option specifies a bump multiplier"""
        self._options.bm = float(next(self._gen))

    def parse_boost(self):
        """The -boost option increases the sharpness, or clarity, of mip-mapped 
        texture files
        """
        self._options.boost = float(next(self._gen))

    def parse_cc(self):
        """The -cc option turns on color correction for the texture"""
        self._options.cc = next(self._gen)

    def parse_clamp(self):
        """The -clamp option turns clamping on or off."""
        self._options.clamp = next(self._gen)

    def parse_imfchan(self):
        """The -imfchan option specifies the channel used to create a scalar or 
        bump texture.
        """
        self._options.imfchan = next(self._gen)

    def parse_mm(self):
        """The -mm option modifies the range over which scalar or color texture 
        values may vary
        """
        base = float(next(self._gen))
        gain = float(next(self._gen))
        self._options.mm = base, gain

    def parse_o(self):
        """The -o option offsets the position of the texture map on the surface by 
        shifting the position of the map origin.
        """
        u = float(next(self._gen))
        v = float(next(self._gen))
        w = float(next(self._gen))
        self._options.o = u, v, w

    def parse_s(self):
        """The -s option scales the size of the texture pattern on the textured 
        surface by expanding or shrinking the pattern
        """
        u = float(next(self._gen))
        v = float(next(self._gen))
        w = float(next(self._gen))
        self._options.s = u, v, w

    def parse_t(self):
        """The -t option turns on turbulence for textures."""
        u = float(next(self._gen))
        v = float(next(self._gen))
        w = float(next(self._gen))
        self._options.t = u, v, w

    def parse_texres(self):
        """The -texres option specifies the resolution of texture created when an 
        image is used.
        """
        self._options.texres = next(self._gen)


class Texture:
    def __init__(self, name, search_path):
        """Create a texture.

        Args:
            name (str): The texture name possibly with path and options as it appear in the material
            search_path (str): Absolute or relative path the texture might be located.
        """
        # The parsed name from the material might contain options
        self._options = TextureOptionsParser(name).parse()
        self._name = self._options.name
        self._search_path = Path(search_path)
        self._path = Path(search_path, self._name)

        # Unsed externally by visualization
        self.image = None

    @property
    def name(self):
        """str: The texture path as it appears in the material"""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def options(self) -> TextureOptions:
        """TextureOptions: Options for this texture"""
        return self._options

    def find(self, path=None):
        """Find the texture in the configured search path
        By default a search will be done in the same directory as
        the obj file including all subdirectories if ``path`` does not exist.

        Args:
            path: Override the search path
        Raises:
            FileNotFoundError if not found
        """
        if self.exists():
            return self.path

        search_path = path or self._search_path
        locations = Path(search_path).glob('**/{}'.format(self.file_name))
        # Attempt to look up the first entry of the generator
        try:
            first = next(locations)
        except StopIteration:
            raise FileNotFoundError("Cannot locate texture `{}` in search path: {}".format(
                self._name, search_path))

        return str(first)

    @property
    def file_name(self):
        """str: Obtains the file name of the texture.
        Sometimes materials contains a relative or absolute path
        to textures, something that often doesn't reflect the
        textures real location.
        """
        if ':' in self._name or '\\' in self._name:
            return PureWindowsPath(self._name).name

        return Path(self._name).name

    @property
    def path(self):
        """str: search_path + name"""
        return str(self._path)

    @path.setter
    def path(self, value):
        self._path = Path(value)

    @property
    def image_name(self):
        """Wrap the old property name to not break compatibility.
        The value will always be the texture path as it appears in the material.
        """
        return self._name

    @image_name.setter
    def image_name(self, value):
        """Wrap the old property name to not break compatibility"""
        self._name = value

    def exists(self):
        """bool: Does the texture exist"""
        return self._path.exists()
