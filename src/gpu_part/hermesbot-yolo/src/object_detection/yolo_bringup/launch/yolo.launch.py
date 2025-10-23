# Copyright (C) 2023 Miguel Ángel González Santamarta
# Modifications (C) 2025 Artem Voronov
# GPL-3.0-or-later

import yaml
from launch import LaunchDescription, LaunchContext, logging
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cfg_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("yolo_bringup"), "config", "yolo_params.yaml"]
        ),
        description="YAML file with all YOLO parameters",
    )
    cfg_path = LaunchConfiguration("config")

    weights_dir_arg = DeclareLaunchArgument(
        "weights_dir",
        default_value="",
        description="Folder with fine-tuned YOLO weights (optional)",
    )
    weights_dir_cfg = LaunchConfiguration("weights_dir")

    def setup(context: LaunchContext, *_):
        path = context.perform_substitution(cfg_path)

        with open(path, "r") as f:
            params_all = yaml.safe_load(f)

        # --- accept either "/**":{ros__parameters:{…}} or legacy "yolo_ros":{…}
        if "/**" in params_all and "ros__parameters" in params_all["/**"]:
            cfg = params_all["/**"]["ros__parameters"]
        else:
            cfg = params_all

        # === PRINT LOADED PARAMS ===
        log = logging.get_logger("yolo_launch")
        log.info(f"Loaded parameters from: {path}")
        for k, v in cfg.items():
            log.info(f"  {k}: {v}")
        # =========================================================

        ns        = cfg.get("namespace", "yolo")
        use_detect = bool(cfg.get("use_detect", True))
        use_track = bool(cfg.get("use_tracking", False))
        use_debug = bool(cfg.get("use_debug", False))

        nodes = []

        if use_detect:
            nodes.append(
                Node(
                    package="yolo_ros", executable="yolo_node",name="yolo_node",
                    namespace=ns, parameters=[path,
                        {"weights_dir": context.perform_substitution(weights_dir_cfg)},
                    ],
                )
            )

        if use_track:
            nodes.append(
                Node(
                    package="yolo_ros", executable="tracking_node", name="tracking_node",
                    namespace=ns, parameters=[path], 
                )
            )

        if use_debug:
            # Choose which detections to visualize: raw detections or tracking output
            detections_topic = "tracking" if use_track else "detections"
            nodes.append(
                Node(
                    package="yolo_ros", executable="debug_node", name="debug_node",
                    namespace=ns, parameters=[
                        path, {"detections_topic": detections_topic}
                    ],
                )
            )
        return nodes

    return LaunchDescription([cfg_arg, weights_dir_arg, OpaqueFunction(function=setup)])
