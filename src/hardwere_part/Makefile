# Export the display environment for GUI applications
export HOST_IP=192.168.50.9
export DISPLAY=:0

# Allow Docker GUI apps to use host X server
prepare_terminal_for_visualization:
	@echo "Preparing X11 display access for Docker..."
	@xhost +local:root

# Build Docker image
build:
	@echo "Building Docker images..."
	docker compose -f docker-compose.yaml build

# Run the container interactively (with roscore in background)
run:
	@echo "Running interactive container with ROS environment..."
	docker compose up
