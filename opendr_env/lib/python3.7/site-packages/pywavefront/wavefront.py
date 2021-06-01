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
import logging
from pywavefront import ObjParser
import pathlib
import pywavefront

logger = logging.getLogger("pywavefront")


class Wavefront:
    # Can be used to override the parser when extending the class
    parser_cls = ObjParser

    """Import a wavefront .obj file."""
    def __init__(
        self,
        file_name,
        strict=False,
        encoding="utf-8",
        create_materials=False,
        collect_faces=False,
        parse=True,
        cache=False,
    ):
        """
        Create a Wavefront instance
        :param file_name: file name and path of obj file to read
        :param strict: Enable strict mode
        :param encoding: What text encoding the parser should use
        :param create_materials: Create materials if they don't exist
        :param parse: Should parse be called immediately or manually called later?
        """
        self.file_name = file_name
        self.mtllibs = []
        self.materials = {}
        self.meshes = {}        # Name mapping
        self.vertices = []
        self.mesh_list = []     # Also includes anonymous meshes

        self.parser = self.parser_cls(
            self,
            self.file_name,
            strict=strict,
            encoding=encoding,
            create_materials=create_materials,
            collect_faces=collect_faces,
            parse=parse,
            cache=cache)

    def parse(self):
        """Manually call the parser. This is used when parse=False"""
        self.parser.parse()

    def add_mesh(self, the_mesh):
        self.mesh_list.append(the_mesh)
        self.meshes[the_mesh.name] = the_mesh
