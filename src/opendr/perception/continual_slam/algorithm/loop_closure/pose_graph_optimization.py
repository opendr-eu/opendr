import g2o
import numpy as np


class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        self.edge_vertices = set()
        self.num_loop_closures = 0

        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        # See https://github.com/RainerKuemmerle/g2o/issues/34
        self.se3_offset_id = 0
        se3_offset = g2o.ParameterSE3Offset()
        se3_offset.set_id(self.se3_offset_id)
        super().add_parameter(se3_offset)

    def __str__(self):
        string = f'Vertices: {len(self.vertex_ids)}\n'
        string += f'Edges:   {len(self.edge_vertices)}\n'
        string += f'Loops:   {self.num_loop_closures}'
        return string

    @property
    def vertex_ids(self):
        return sorted(list(self.vertices().keys()))

    def optimize(self, max_iterations=1000, verbose=False):
        super().initialize_optimization()
        super().set_verbose(verbose)
        super().optimize(max_iterations)

    def add_vertex(self, vertex_id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(vertex_id)
        v_se3.set_estimate(g2o.Isometry3d(pose))
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_vertex_point(self, vertex_id, point, fixed=False):
        v_point = g2o.VertexPointXYZ()
        v_point.set_id(vertex_id)
        v_point.set_estimate(point)
        v_point.set_fixed(fixed)
        super().add_vertex(v_point)

    def add_edge(self,
                 vertices,
                 measurement,
                 information=np.eye(6),
                 robust_kernel=None,
                 is_loop_closure=False):
        self.edge_vertices.add(vertices)
        if is_loop_closure:
            self.num_loop_closures += 1

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(g2o.Isometry3d(measurement))  # relative pose
        edge.set_information(information)
        # robust_kernel = g2o.RobustKernelHuber()
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def add_edge_pose_point(self,
                            vertex_pose,
                            vertex_point,
                            measurement,
                            information=np.eye(3),
                            robust_kernel=None):
        edge = g2o.EdgeSE3PointXYZ()
        edge.set_vertex(0, self.vertex(vertex_pose))
        edge.set_vertex(1, self.vertex(vertex_point))
        edge.set_measurement(measurement)
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        edge.set_parameter_id(0, self.se3_offset_id)
        super().add_edge(edge)

    def get_pose(self, vertex_id):
        return self.vertex(vertex_id).estimate().matrix()

    def get_all_poses(self):
        return [self.get_pose(i) for i in self.vertex_ids]

    def get_transform(self, vertex_id_src, vertex_id_dst):
        pose_src = self.get_pose(vertex_id_src)
        pose_dst = self.get_pose(vertex_id_dst)
        transform = np.linalg.inv(pose_src) @ pose_dst
        return transform

    def does_edge_exists(self, vertex_id_a, vertex_id_b):
        return (vertex_id_a,
                vertex_id_b) in self.edge_vertices or (vertex_id_b,
                                                       vertex_id_a) in self.edge_vertices

    def is_vertex_in_any_edge(self, vertex_id):
        vertices = set()
        for edge in self.edge_vertices:
            vertices.add(edge[0])
            vertices.add(edge[1])
        return vertex_id in vertices

    def does_vertex_have_only_global_edges(self, vertex_id):
        assert self.is_vertex_in_any_edge(vertex_id)
        for edge in self.edge_vertices:
            if vertex_id not in edge:
                continue
            if np.abs(edge[0] - edge[1]) == 1:
                return False
        return True

    def return_last_positions(self, n=20, reversed=False):
        positions = []
        keys = list(self.vertices().keys())
        length = len(keys)
        i = 0
        while i < n and i < length:
            if reversed:
                key = keys[length-i-1]
            else:
                key = keys[i]
            vertex = self.vertices()[key]
            if isinstance(vertex, g2o.VertexSE3):
                positions.append(vertex.estimate().matrix()[:3, 3])
            i += 1
        return positions
    
    def return_all_positions(self):
        positions = []
        for _, vertex in self.vertices().items():
            if isinstance(vertex, g2o.VertexSE3):
                positions.append(vertex.estimate().matrix()[:3, 3])
        return list(reversed(positions))

    def return_last_poses(self, n=20, reversed=False):
        poses = []
        keys = list(self.vertices().keys())
        length = len(keys)
        i = 0
        while i < n and i < length:
            if reversed:
                key = keys[length-i-1]
            else:
                key = keys[i]
            vertex = self.vertices()[key]
            if isinstance(vertex, g2o.VertexSE3):
                poses.append(vertex.estimate().matrix())
            i += 1
        return poses