import os

import numpy as np

from util import ACTION_FILE_NAMES


def parse_skeleton_text(line):
    # Parse line by comma
    fields = line.split(',')
    # assert len(fields) == 172, 'Actual length is: ' + str(len(fields))

    frame_num = fields[0]
    skeleton_coords = []

    offset = 1
    for joint_id in range(1, 16):  # 1, 2,...,11
        if joint_id <= 11:  # 1, 2,...,11
            offset += 10  # skip orientation and conf

        x = float(fields[offset])
        offset += 1
        y = float(fields[offset])
        offset += 1
        z = float(fields[offset])
        offset += 1
        conf = float(fields[offset])
        offset += 1

        # pixel_x, pixel_y = pixel_from_coords(x, y, z)
        skeleton_coords.append([x, y, z])

    # assert len(skeleton_coords) == 15, 'Actual length is: ' + str(len(output))
    skeleton_coords = [
        skeleton_coords[1],
        skeleton_coords[0],
        skeleton_coords[3],
        skeleton_coords[4],
        skeleton_coords[11],
        skeleton_coords[5],
        skeleton_coords[6],
        skeleton_coords[12],
        skeleton_coords[8],
        skeleton_coords[13],
        skeleton_coords[10],
        skeleton_coords[14],
        skeleton_coords[7],
        skeleton_coords[9],
        skeleton_coords[2],
    ]

    skeleton_coords = np.array(skeleton_coords)
    return frame_num, skeleton_coords  # ",".join((str(v) for v in output))


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
