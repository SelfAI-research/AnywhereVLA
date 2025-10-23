# auto-pick platform: jetson if L4T detected, else nuc
PLATFORM ?= $(shell sh -c '\
  if [ -f /etc/nv_tegra_release ]; then echo jetson; \
  elif command -v dpkg >/dev/null 2>&1 && dpkg -s nvidia-l4t-core >/dev/null 2>&1; then echo jetson; \
  else echo nuc; fi')

COMPOSE_BASE := docker-compose.yml
COMPOSE_DIR  := docker.$(PLATFORM)
COMPOSE_OVERRIDE := $(COMPOSE_DIR)/compose.$(PLATFORM).yml
DC := docker compose -f $(COMPOSE_BASE) -f $(COMPOSE_OVERRIDE)

.PHONY: build yolo_ros attach rviz clean vars

build:
	$(DC) build

yolo_ros:
	$(DC) up yolo_ros

attach:
	$(DC) run --rm --remove-orphans enter

rviz:
	$(DC) up rviz -d

clean:
	$(DC) down -v

vars:
	@echo "PLATFORM=$(PLATFORM)"
	@echo "COMPOSE_OVERRIDE=$(COMPOSE_OVERRIDE)"
	@ls -l $(COMPOSE_OVERRIDE) || true
