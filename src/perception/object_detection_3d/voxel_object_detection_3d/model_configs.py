TANET_16 = {
    "voxel_generator": {
        "point_cloud_range": [0, -39.68, -3, 69.12, 39.68, 1],
        "voxel_size": [0.16, 0.16, 4],
        "max_number_of_points_per_voxel": 100,
    },
    "num_class": 1,
    "voxel_feature_extractor": {
        "module_class_name": "PillarFeature_TANet",
        "num_filters": [64],
        "with_distance": False,
    },
    "middle_feature_extractor": {"module_class_name": "PointPillarsScatter",},
    "rpn": {
        "module_class_name": "PSA",
        "layer_nums": [3, 5, 5],
        "layer_strides": [2, 2, 2],
        "num_filters": [64, 128, 256],
        "upsample_strides": [1, 2, 4],
        "num_upsample_filters": [128, 128, 128],
        "use_groupnorm": False,
        "num_groups": 32,
    },
    "loss": {
        "classification_loss": {
            "weighted_sigmoid_focal": {
                "alpha": 0.25,
                "gamma": 2.0,
                "anchorwise_output": True,
            },
        },
        "localization_loss": {
            "weighted_smooth_l1": {
                "sigma": 3.0,
                "code_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        },
        "classification_weight": 1.0,
        "localization_weight": 2.0,
    },
    "use_sigmoid_score": True,
    "encode_background_as_zeros": True,
    "encode_rad_error_by_sin": True,
    "use_direction_classifier": True,
    "direction_loss_weight": 0.2,
    "use_aux_classifier": False,
    # Loss
    "pos_class_weight": 1.0,
    "neg_class_weight": 1.0,
    "loss_norm_type": "NormByNumPositives",
    # Postprocess
    "post_center_limit_range": [0, -39.68, -5, 69.12, 39.68, 5],
    "use_rotate_nms": False,
    "use_multi_class_nms": False,
    "nms_pre_max_size": 1000,
    "nms_post_max_size": 300,
    "nms_score_threshold": 0.3,  # 0.05
    "nms_iou_threshold": 0.1,  # 0.5
    "use_bev": False,
    "num_point_features": 4,
    "without_reflectivity": False,
    "box_coder": {
        "ground_box3d_coder": {
            "linear_dim": False,
            "encode_angle_vector": False,
        },
    },
    "target_assigner": {
        "anchor_generators": {
            "anchor_generator_stride": {
                "sizes": [1.6, 3.9, 1.56],  # wlh
                "strides": [
                    0.32,
                    0.32,
                    0.0,
                ],  # if generate only 1 z_center, z_stride will be ignored
                "offsets": [
                    0.16,
                    -39.52,
                    -1.78,
                ],  # origin_offset + strides / 2
                "rotations": [0, 1.57],  # 0, pi/2
                "matched_threshold": 0.6,
                "unmatched_threshold": 0.45,
            },
        },
        "sample_positive_fraction": -1,
        "sample_size": 512,
        "region_similarity_calculator": {"nearest_iou_similarity": {},},
    },
}


backbones = {
    "tanet_16": TANET_16
}
