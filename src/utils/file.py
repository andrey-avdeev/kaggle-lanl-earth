from os import listdir
from os.path import isfile, join
from typing import List


def file_names(folder: str) -> List[str]:
    names = [f for f in listdir(folder) if isfile(join(folder, f))]

    return names
