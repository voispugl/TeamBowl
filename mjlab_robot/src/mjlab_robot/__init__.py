"""TeamBowl mjlab robot package."""

from mjlab_robot.tasks import balance  # noqa: F401  (side-effect: registers tasks)


def main() -> None:
    print("TeamBowl mjlab tasks loaded.")
