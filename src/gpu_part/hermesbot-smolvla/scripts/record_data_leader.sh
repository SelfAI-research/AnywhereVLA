HF_USER=VorArt
FOLLOWER_PORT=/dev/ttyCH341USB1
LEADER_PORT=/dev/ttyCH341USB0
DATASET_NAME=hermesbot_dataset_3_plactic_cup
DATASET_ROOT="/workspace/hf_home/lerobot/${HF_USER}/${DATASET_NAME}"
RESUME=true

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=${FOLLOWER_PORT} \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=${LEADER_PORT} \
  --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{
    wrist: {type: intelrealsense, serial_number_or_name: 049322070172, width: 640, height: 360, fps: 15, warmup_s: 3},
    base:  {type: intelrealsense, serial_number_or_name: 934222071152, width: 640, height: 360, fps: 15, warmup_s: 3},
    side:  {type: intelrealsense, serial_number_or_name: 135122071874, width: 640, height: 360, fps: 15, warmup_s: 3}
  }" \
  --dataset.single_task="Pick up the plactic cup and place it into the blue box" \
  --dataset.repo_id=${HF_USER}/${DATASET_NAME} \
  --dataset.root=${DATASET_ROOT} \
  --dataset.num_episodes=25 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=2 \
  --dataset.fps=15 \
  --dataset.num_image_writer_threads_per_camera=8 \
  --dataset.num_image_writer_processes=2 \
  --dataset.video=false \
  --display_data=false \
  --play_sounds=false \
  --resume=${RESUME}
