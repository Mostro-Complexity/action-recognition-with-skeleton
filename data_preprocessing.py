import os
import numpy as np

from util.CAD_60 import ACTION_FILE_NAMES, parse_skeleton_text

ORIGINAL_DATA_PATH = 'data/original'
INPUT_DATA_PATH = 'data/input'

if __name__ == "__main__":
    filenames = ['.'.join([fname, 'txt'])
                 for fname in ACTION_FILE_NAMES.keys()]

    for fn in filenames:
        with open(os.path.join(ORIGINAL_DATA_PATH, fn), 'r') as f:
            content = [line.rstrip() for line in f.readlines()]
            content.pop()

            all_coords = np.array([parse_skeleton_text(
                line)[1] for line in content])

            video_id = os.path.splitext(fn)[0]

            np.save(os.path.join(INPUT_DATA_PATH, video_id), all_coords)
