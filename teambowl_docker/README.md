# TeamBowl Docker

ROS2 Humble container for the TeamBowl robot. On first run it automatically builds the workspace, then launches the full system via `bringup.launch.py`.

## Requirements

- Docker + Docker Compose installed on the host
- Run from the `teambowl_docker/` directory

## Build the image

Only needed once (or after changing the Dockerfile):

```bash
docker compose build
```

## Run the robot

```bash
docker compose up
```

- **First run:** builds the ROS2 workspace (~5–10 min), then launches all nodes automatically.
- **Subsequent runs:** skips the build and launches immediately.

## Stop the robot

```bash
docker compose down
```

Or press `Ctrl+C` if running in the foreground.

## Get a debug shell (instead of auto-launching)

```bash
docker compose run --rm teambowl bash
```

## Attach a shell to a running container

```bash
docker exec -it teambowl_dev bash
```

## Force a workspace rebuild

Delete the build marker, then restart:

```bash
docker exec teambowl_dev rm /workspaces/teambowl_ws/install/.colcon_build_complete
docker compose restart
```

## View build logs

If the workspace build fails, logs are saved inside the container:

```bash
docker exec teambowl_dev cat /tmp/colcon_build.log
```
