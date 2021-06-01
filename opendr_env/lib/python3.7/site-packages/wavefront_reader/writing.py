import numpy as np

def grouper(n, iterable):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip(*[iter(iterable)]*n)


class WavefrontWriter(object):

    preamble = "# Blender v2.69 (sub 5) OBJ File: ''\n" + "# www.blender.org\n"

    def __init__(self, string):
        self._data = self.preamble + string


    @classmethod
    def from_dicts(cls, mesh_name, vert_dict, normal_dict):
        """Returns a wavefront .obj string using pre-triangulated vertex dict and normal dict as reference."""

        # Put header in string
        wavefront_str = "o {name}\n".format(name=mesh_name)

        # Write Vertex data from vert_dict
        for wall in vert_dict:
            for vert in vert_dict[wall]:
                wavefront_str += "v {0} {1} {2}\n".format(*vert)

        # Write (false) UV Texture data
        wavefront_str += "vt 1.0 1.0\n"

        # Write Normal data from normal_dict
        for wall, norm in normal_dict.items():
            wavefront_str += "vn {0} {1} {2}\n".format(*norm)

        # Write Face Indices (1-indexed)
        vert_idx = 0
        for wall in vert_dict:
            for _ in range(0, len(vert_dict[wall]), 3):
                wavefront_str += 'f '
                for vert in range(3): # 3 vertices in each face
                    vert_idx += 1
                    wavefront_str += "{v}/1/{n} ".format(v=vert_idx, n=wall+1)
                wavefront_str = wavefront_str[:-1] + '\n'  # Cutoff trailing space and add a newline.

        # Return Wavefront string
        return WavefrontWriter(string=wavefront_str)

    @classmethod
    def from_arrays(cls, name, verts, normals, n_verts=3):

        """Returns a wavefront .obj string using pre-triangulated vertex dict and normal dict as reference."""

        # Put header in string
        wavefront_str = "o {name}\n".format(name=name)

        # Write Vertex data from vert_dict
        for vert in verts:
            wavefront_str += "v {0} {1} {2}\n".format(*vert)

        # Write (false) UV Texture data
        wavefront_str += "vt 1.0 1.0\n"

        for norm in normals:
            wavefront_str += "vn {0} {1} {2}\n".format(*norm)


        # Write Face Indices (1-indexed)
        for wall_idx, vert_indices in enumerate(grouper(n_verts, range(1, len(verts) + 1))):
            face_str = 'f '
            for idx in vert_indices:
                face_str += "{v}/1/{n} ".format(v=idx, n=wall_idx + 1)
            wavefront_str += face_str + '\n'

        return WavefrontWriter(string=wavefront_str)

    @classmethod
    def from_indexed_arrays(cls, name, verts, normals, face_indices):
        """Takes MxNx3 verts, Mx3 normals to build obj file"""

        # Put header in string
        wavefront_str = "o {name}\n".format(name=name)

        assert face_indices.ndim == 2

        # Write Vertex data from vert_dict
        for vert in verts:
            wavefront_str += "v {0} {1} {2}\n".format(*vert)

        # Write (false) UV Texture data
        wavefront_str += "vt 1.0 1.0\n"

        for norm in normals:
            wavefront_str += "vn {0} {1} {2}\n".format(*norm)

        assert len(face_indices) == len(normals) * 2
        for norm_idx, vert_idx,  in enumerate(face_indices):
            wavefront_str += "f"
            for vv in vert_idx:
                wavefront_str += " {}/{}/{}".format(vv + 1, 1, (norm_idx // 2) + 1 )
            wavefront_str += "\n"

        return cls(string=wavefront_str)

    def dump(self, f):
        """Write Wavefront data to file.  Takes File object or filename."""
        try:
            f.write(self._data)
        except AttributeError:
            with open(f, 'w') as wf:
                wf.write(self._data)


    def dumps(self):
        """Return Wavefront-formatted data as a string"""
        return self._data
