#  Final
HF_USER=$(huggingface-cli whoami | head -n 1)
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=<LEADER_PORT> \
  --teleop.id=my_awesome_leader_arm \
  --robot.cameras="{
    wrist: {type: intelrealsense, serial_number_or_name: 049322070172, width: 640, height: 480, fps: 15},
    base:  {type: intelrealsense, serial_number_or_name: 934222071152, width: 640, height: 480, fps: 15},
    side:  {type: intelrealsense, serial_number_or_name: 135122071874, width: 640, height: 480, fps: 15}
  }" \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/hermesbot_pick_bottle \
  --dataset.single_task="Pick the bottle" \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=40 \
  --dataset.reset_time_s=40 \
  --dataset.video=true

# To just run VLA
FOLLOWER_PORT=ttyCH341USB0


lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyCH341USB0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{
    image: {type: intelrealsense, serial_number_or_name: 049322070172, width: 640, height: 480, fps: 30},
    image2:  {type: intelrealsense, serial_number_or_name: 934222071152, width: 640, height: 480, fps: 30},
    image3:  {type: intelrealsense, serial_number_or_name: 135122071874, width: 640, height: 480, fps: 30}
  }" \
  --dataset.repo_id=VorArt/eval_test16 \
  --dataset.push_to_hub=false \
  --dataset.single_task="pick up the bottle" \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --policy.path=/workspace/model/smolvla_base \
  --dataset.video=true \
  --policy.device=cuda \
  --display_data=true \
  --play_sounds=false 
