# Copyright 2020-2021 OpenDR European Project
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

# model settings
model = dict(
	type='EfficientLPS',
	pretrained=True,
	backbone=dict(
		type='tf_efficientnet_b5',
		act_cfg=dict(type="Identity"),
		norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
		in_chans=5,
		style='pytorch'),
	neck=dict(
		type='RangeAwareFPN',
		in_channels=[40, 64, 176, 2048],  # b0[24, 40, 112, 1280], #b4[32, 56, 160, 1792],
		out_channels=256,
		norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
		act_cfg=None,
		num_outs=4),
	rpn_head=dict(
		type='RPNHead',
		in_channels=256,
		feat_channels=256,
		anchor_scales=[8],
		anchor_ratios=[0.5, 1.0, 2.0],
		anchor_strides=[4, 8, 16, 32],
		target_means=[.0, .0, .0, .0],
		target_stds=[1.0, 1.0, 1.0, 1.0],
		loss_cls=dict(
			type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
		loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
	bbox_roi_extractor=dict(
		type='SingleRoIExtractor',
		roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
		out_channels=256,
		featmap_strides=[4, 8, 16, 32]),
	bbox_head=dict(
		type='SharedFCBBoxHead',
		num_fcs=2,
		in_channels=256,
		fc_out_channels=1024,
		roi_feat_size=7,
		num_classes=9,
		target_means=[0., 0., 0., 0.],
		target_stds=[0.1, 0.1, 0.2, 0.2],
		norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
		reg_class_agnostic=False,
		loss_cls=dict(
			type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
		loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
	mask_roi_extractor=dict(
		type='SingleRoIExtractor',
		roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
		out_channels=256,
		featmap_strides=[4, 8, 16, 32]),
	mask_head=dict(
		type='FCNSepMaskHead',
		num_convs=4,
		in_channels=256,
		conv_out_channels=256,
		num_classes=9,
		norm_cfg=dict(type='InPlaceABN', activation='leaky_relu', activation_param=0.01, requires_grad=True),
		act_cfg=None,
		loss_mask=dict(
			type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
	semantic_head=dict(
		type='EfficientLPSSemanticHead',
		in_channels=256,
		conv_out_channels=128,
		num_classes=19,
		ignore_label=255,
		loss_weight=1.0,
		ohem=0.25,
		norm_cfg=dict(type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01, requires_grad=True),
		act_cfg=None))

# model training and testing settings
train_cfg = dict(
	rpn=dict(
		assigner=dict(
			type='MaxIoUAssigner',
			pos_iou_thr=0.7,
			neg_iou_thr=0.3,
			min_pos_iou=0.3,
			ignore_iof_thr=-1),
		sampler=dict(
			type='RandomSampler',
			num=256,
			pos_fraction=0.5,
			neg_pos_ub=-1,
			add_gt_as_proposals=False),
		allowed_border=0,
		pos_weight=-1,
		debug=False),
	rpn_proposal=dict(
		nms_across_levels=False,
		nms_pre=2000,
		nms_post=2000,
		max_num=2000,
		nms_thr=0.7,
		min_bbox_size=0),
	rcnn=dict(
		assigner=dict(
			type='MaxIoUAssigner',
			pos_iou_thr=0.5,
			neg_iou_thr=0.5,
			min_pos_iou=0.5,
			ignore_iof_thr=-1),
		sampler=dict(
			type='RandomSampler',
			num=256,
			pos_fraction=0.25,
			neg_pos_ub=-1,
			add_gt_as_proposals=True),
		mask_size=28,
		pos_weight=-1,
		debug=False))

test_cfg = dict(
	rpn=dict(
		nms_across_levels=False,
		nms_pre=1000,
		nms_post=1000,
		max_num=1000,
		nms_thr=0.7,
		min_bbox_size=0),
	rcnn=dict(
		score_thr=0.0,
		nms=dict(type='nms', iou_thr=0.5),
		max_per_img=100,
		mask_thr_binary=0.5),
	panoptic=dict(
		overlap_thr=0.1,
		min_stuff_area=4096)
)

# dataset settings
# dataset_type = 'SemanticKITTIDataset'  # TODO: Remove these two lines? (As per diff ÷ efficientps/algorithm/config and efficientps/config in Niclas' code)
# data_root = '/home/mohan/mot_challenge/lidar_track/epsnet/scripts/kalman/'  # TODO: Set correct path
train_pipeline = [
	dict(type='LoadLidarFromFile', project=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0, gt=True, max_points=150000,
		 sensor_img_means=[12.12, 10.88, 0.23, -1.04, 0.21], sensor_img_stds=[12.32, 11.47, 6.91, 0.86, 0.16]),
	dict(type='Resize', img_scale=(4096, 256), multiscale_mode='value', keep_ratio=False),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]

test_pipeline = [
	dict(type='LoadLidarFromFile', project=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0, gt=True, max_points=150000,
		 sensor_img_means=[12.12, 10.88, 0.23, -1.04, 0.21], sensor_img_stds=[12.32, 11.47, 6.91, 0.86, 0.16]),
	dict(type='Resize', img_scale=(4096, 256), multiscale_mode='value', keep_ratio=False),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img']),
]

# data = dict(  # TODO: Remove this whole block ? (As per diff ÷ efficientps/algorithm/config and efficientps/config in Niclas' code)
# 	imgs_per_gpu=3,  # TODO: Check value (as per diff ÷ efficientlps/alg/config/singlegpu and multigpu)
# 	workers_per_gpu=1,  # TODO: Chech value
# 	train=dict(
# 		type=dataset_type,
# 		ann_file=data_root+'sequences',
# 		config='configs/semantic-kitti.yaml',
# 		split='train',
# 		pipeline=train_pipeline),
# 	val=dict(
# 		type=dataset_type,
# 		ann_file=data_root+'sequences',
# 		config='configs/semantic-kitti.yaml',
# 		split='valid',
# 		pipeline=test_pipeline),
# 	test=dict(
# 		type=dataset_type,
# 		ann_file=data_root+'sequences',
# 		config='configs/semantic-kitti.yaml',
# 		split='valid',
# 		pipeline=test_pipeline)
# )
#
# evaluation = dict(interval=1, metric=['panoptic'])
#
# # optimizer
# optimizer = dict(type='SGD', lr=0.07, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
#
# # learning policy
# lr_config = dict(
# 	policy='step',
# 	warmup='linear',
# 	warmup_iters=500,
# 	warmup_ratio=1.0 / 3,
# 	step=[120, 144])
# checkpoint_config = dict(interval=1)
# # yapf:disable
# log_config = dict(
# 	interval=1,
# 	hooks=[
# 		dict(type='TextLoggerHook'),
# 		dict(type='TensorboardLoggerHook')
# 	])
# # yapf:enable
# # runtime settings
# total_epochs = 160
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# work_dir = None
# load_from = None
# resume_from = None
# workflow = [('train', 1)]
