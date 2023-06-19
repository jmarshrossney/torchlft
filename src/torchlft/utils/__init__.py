from enum import Enum


def raise_(exc: Exception):
    raise Exception


# NOTE: StrEnum coming in Python 3.11
class StrEnum(str, Enum):
    def __str__(self):
        return str(self.name)
