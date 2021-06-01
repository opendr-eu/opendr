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
from collections import namedtuple
import logging
import time

from pywavefront.exceptions import PywavefrontException
from pywavefront.parser import Parser, auto_consume
from pywavefront.material import Material, MaterialParser
from pywavefront.mesh import Mesh
from pywavefront.cache import Meta, CacheWriter, CacheLoader

logger = logging.getLogger("pywavefront")


class ObjParser(Parser):
    """This parser parses lines from .obj files."""
    material_parser_cls = MaterialParser
    cache_loader_cls = CacheLoader
    cache_writer_cls = CacheWriter

    def __init__(self, wavefront, file_name, strict=False, encoding="utf-8",
                 create_materials=False, collect_faces=False, parse=True, cache=False):
        """
        Create a new obj parser
        :param wavefront: The wavefront object
        :param file_name: file name and path of obj file to read
        :param strict: Enable strict mode
        :param encoding: Encoding to read the text files
        :param create_materials: Create materials if they don't exist
        :param cache: Cache the loaded obj files in binary format
        :param parse: Should parse be called immediately or manually called later?
        """
        super(ObjParser, self).__init__(file_name, strict=strict, encoding=encoding)
        self.wavefront = wavefront

        self.mesh = None
        self.material = None
        self.create_materials = create_materials
        self.collect_faces = collect_faces
        self.cache = cache
        self.cache_loaded = None

        # Stores normals and texcoords for the entire file
        self.normals = []
        self.tex_coords = []

        if parse:
            self.parse()

    def parse(self):
        """Trigger cache load or call superclass parse()"""
        start = time.time()

        if self.cache:
            self.load_cache()

        if not self.cache_loaded:
            super(ObjParser, self).parse()

        logger.info("%s: Load time: %s", self.file_name, time.time() - start)

    def load_cache(self):
        """Loads the file using cached data"""
        self.cache_loaded = self.cache_loader_cls(
            self.file_name,
            self.wavefront,
            strict=self.strict,
            create_materials=self.create_materials,
            encoding=self.encoding,
            parse=self.parse,
        ).parse()

    def post_parse(self):
        """Called after parsing is done"""
        if self.cache and not self.cache_loaded:
            self.cache_writer_cls(self.file_name, self.wavefront).write()

    # methods for parsing types of wavefront lines
    def parse_v(self):
        self.wavefront.vertices += list(self.consume_vertices())

    def consume_vertices(self):
        """
        Consumes all consecutive vertices.
        NOTE: There is no guarantee this will consume all vertices since other
        statements can also occur in the vertex list
        """
        while True:
            # Vertex color
            if len(self.values) == 7:
                yield (
                    float(self.values[1]),
                    float(self.values[2]),
                    float(self.values[3]),
                    float(self.values[4]),
                    float(self.values[5]),
                    float(self.values[6]),
                )
            # Positions only
            else:
                yield (
                    float(self.values[1]),
                    float(self.values[2]),
                    float(self.values[3]),
                )

            try:
                self.next_line()
            except StopIteration:
                break

            if not self.values:
                break

            if self.values[0] != "v":
                break

    def parse_vn(self):
        self.normals += list(self.consume_normals())

        # Since list() also consumes StopIteration we need to sanity check the line
        # to make sure the parser advances
        if self.values and self.values[0] == "vn":
            self.next_line()

    def consume_normals(self):
        """Consumes all consecutive texture coordinate lines"""
        # The first iteration processes the current/first vn statement.
        # The loop continues until there are no more vn-statements or StopIteration is raised by generator
        while True:
            yield (
                float(self.values[1]),
                float(self.values[2]),
                float(self.values[3]),
            )

            try:
                self.next_line()
            except StopIteration:
                break

            if not self.values:
                break

            if self.values[0] != "vn":
                break

    def parse_vt(self):
        self.tex_coords += list(self.consume_texture_coordinates())

        # Since list() also consumes StopIteration we need to sanity check the line
        # to make sure the parser advances
        if self.values and self.values[0] == "vt":
            self.next_line()

    def consume_texture_coordinates(self):
        """Consume all consecutive texture coordinates"""
        # The first iteration processes the current/first vt statement.
        # The loop continues until there are no more vt-statements or StopIteration is raised by generator
        while True:
            yield (
                float(self.values[1]),
                float(self.values[2]),
            )

            try:
                self.next_line()
            except StopIteration:
                break

            if not self.values:
                break

            if self.values[0] != "vt":
                break

    @auto_consume
    def parse_mtllib(self):
        mtllib = " ".join(self.values[1:])
        try:
            materials = self.material_parser_cls(
                self.dir / mtllib,
                encoding=self.encoding,
                strict=self.strict,
                collect_faces=self.collect_faces
            ).materials
            self.wavefront.mtllibs.append(mtllib)
        except IOError:
            if self.create_materials:
                return
            raise

        for name, material in materials.items():
            self.wavefront.materials[name] = material

    @auto_consume
    def parse_usemtl(self):
        name = " ".join(self.values[1:])
        self.material = self.wavefront.materials.get(name, None)

        if self.material is None:
            if not self.create_materials:
                raise PywavefrontException('Unknown material: %s' % name)

            # Create a new default material if configured to resolve missing ones
            self.material = Material(name, is_default=True, has_faces=self.collect_faces)
            self.wavefront.materials[name] = self.material

        if self.mesh is not None:
            self.mesh.add_material(self.material)

    def parse_usemat(self):
        self.parse_usemtl()

    @auto_consume
    def parse_o(self):
        self.mesh = Mesh(self.values[1], has_faces=self.collect_faces)
        self.wavefront.add_mesh(self.mesh)

    def parse_f(self):
        # Add default material if not created
        if self.material is None:
            self.material = Material(
                "default{}".format(len(self.wavefront.materials)),
                is_default=True,
                has_faces=self.collect_faces
            )
            self.wavefront.materials[self.material.name] = self.material

        # Support objects without `o` statement
        if self.mesh is None:
            self.mesh = Mesh(has_faces=self.collect_faces)
            self.wavefront.add_mesh(self.mesh)
            self.mesh.add_material(self.material)

        self.mesh.add_material(self.material)

        collected_faces = []
        consumed_vertices = self.consume_faces(collected_faces if self.collect_faces else None)
        self.material.vertices += list(consumed_vertices)

        if self.collect_faces:
            self.mesh.faces += list(collected_faces)

        # Since list() also consumes StopIteration we need to sanity check the line
        # to make sure the parser advances
        if self.values and self.values[0] == "f":
            self.next_line()

    def consume_faces(self, collected_faces = None):
        """
        Consume all consecutive faces

        If more than three vertices are specified, we triangulate by the following procedure:

            Let the face have n vertices in the order v_1 v_2 v_3 ... v_n, n >= 3.
            We emit the first face as usual: (v_1, v_2, v_3). For each remaining vertex v_j,
            j > 3, we emit (v_j, v_1, v_{j - 1}), e.g. (v_4, v_1, v_3), (v_5, v_1, v_4).

        In a perfect world we could consume all vertices straight forward and draw using
        GL_TRIANGLE_FAN (which exactly matches the procedure above).
        This is however rarely the case.

        * If the face is co-planar but concave, then you need to triangulate the face.
        * If the face is not-coplanar, you are screwed, because OBJ doesn't preserve enough information
          to know what tessellation was intended.

        We always triangulate to make it simple.

            :param collected_faces: A list into which all (possibly triangulated) faces will be written in the form
                                    of triples of the corresponding absolute vertex IDs. These IDs index the list
                                    self.wavefront.vertices.
                                    Specify None to prevent consuming faces (and thus saving memory usage).
        """

        # Helper tuple and function
        Vertex = namedtuple('Vertex', 'idx pos color uv normal')
        def emit_vertex(vertex):
            # Just yield all the values except for the index
            for v in vertex.uv:
                yield v

            for v in vertex.color:
                yield v

            for v in vertex.normal:
                yield v

            for v in vertex.pos:
                yield v


        # Figure out the format of the first vertex
        # We raise an exception if any following vertex has a different format
        # NOTE: Order is always v/vt/vn where v is mandatory and vt and vn is optional
        has_vt = False
        has_vn = False
        has_colors = False

        parts = self.values[1].split('/')
        # We assume texture coordinates are present
        if len(parts) == 2:
            has_vt = True

        # We have a vn, but not necessarily a vt
        elif len(parts) == 3:
            # Check for empty vt "1//1"
            if parts[1] != '':
                has_vt = True
            has_vn = True

        # Are we referencing vertex with color info?
        vindex = int(parts[0])
        if vindex < 0:
            vindex += len(self.wavefront.vertices)
        else:
            vindex -= 1

        vertex = self.wavefront.vertices[vindex]
        has_colors = len(vertex) == 6

        # Prepare vertex format string
        vertex_format = "_".join(e[0] for e in [
            ("T2F", has_vt),
            ("C3F", has_colors),
            ("N3F", has_vn),
            ("V3F", True)
        ] if e[1])

        # If the material already have vertex data, ensure the same format is used
        if self.material.vertex_format and self.material.vertex_format != vertex_format:
            raise ValueError((
                "Trying to merge vertex data with different format: {}. "
                "Material {} has vertex format {}"
            ).format(vertex_format, self.material.name, self.material.vertex_format))

        self.material.vertex_format = vertex_format

        # The first iteration processes the current/first f statement.
        # The loop continues until there are no more f-statements or StopIteration is raised by generator
        while True:
            # The very first vertex, the last encountered and the current one
            v1, vlast, vcurrent = None, None, None

            for i, v in enumerate(self.values[1:]):
                parts = v.split('/')
                v_index = (int(parts[0]) - 1)
                # uv field might be blank
                try:
                    t_index = (int(parts[1]) - 1) if has_vt else None
                except ValueError:
                    t_index = 0
                try:
                    n_index = (int(parts[2]) - 1) if has_vn else None
                except ValueError:
                    n_index = 0

                # Resolve negative index lookups
                if v_index < 0:
                    v_index += len(self.wavefront.vertices) + 1

                if has_vt and t_index < 0:
                    t_index += len(self.tex_coords) + 1

                if has_vn and n_index < 0:
                    n_index += len(self.normals) + 1

                vlast = vcurrent
                vcurrent = Vertex(
                    idx = v_index,
                    pos = self.wavefront.vertices[v_index][0:3] if has_colors else self.wavefront.vertices[v_index],
                    color = self.wavefront.vertices[v_index][3:] if has_colors else (),
                    uv = self.tex_coords[t_index] if has_vt and t_index < len(self.tex_coords) else (),
                    normal = self.normals[n_index] if has_vn and n_index < len(self.normals) else ()
                )

                yield from emit_vertex(vcurrent)

                # Triangulation when more than 3 elements are present
                if i >= 3:
                    # The current vertex has already been emitted.
                    # Now just emit the first and the third vertices from the face
                    yield from emit_vertex(v1)
                    yield from emit_vertex(vlast)

                if i == 0:
                    # Store the first vertex
                    v1 = vcurrent

                if (collected_faces is not None) and (i >= 2):
                    if i == 2:
                        # Append the first triangle face in usual order (i.e. as specified in the Wavefront file)
                        collected_faces.append([v1.idx, vlast.idx, vcurrent.idx])
                    if i >= 3:
                        # Triangulate the remaining part of the face by putting the current, the first
                        # and the last parsed vertex in that order as a new face.
                        # This order coincides deliberately with the order from vertex yielding above.
                        collected_faces.append([vcurrent.idx, v1.idx, vlast.idx])

            # Break out of the loop when there are no more f statements

            try:
                self.next_line()
            except StopIteration:
                break

            if not self.values:
                break

            if self.values[0] != "f":
                break
