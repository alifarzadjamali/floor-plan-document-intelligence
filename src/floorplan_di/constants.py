from enum import IntEnum


class PlanClass(IntEnum):
    BACKGROUND = 0
    ROOM = 1
    WALL = 2
    DOOR = 3
    WINDOW = 4


CLASS_NAMES = tuple(member.name.lower() for member in PlanClass)
CLASS_COLOURS_RGB = (
    (245, 245, 245),
    (91, 192, 222),
    (45, 45, 45),
    (240, 173, 78),
    (92, 184, 92),
)
