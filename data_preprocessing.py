import os

import numpy as np


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


ACTION_FILE_NAMES = {
    '0510160858': 'still',
    '0510161326': 'talking on the phone',
    '0510165136': 'writing on whiteboard',
    '0510161658': 'drinking water',
    '0510171120': 'rinsing mouth with water',
    '0510170707': 'brushing teeth',
    '0510171427': 'wearing contact lenses',
    '0510171507': 'wearing contact lenses',
    '0510162529': 'talking on couch',
    '0510162821': 'relaxing on couch',
    '0510164129': 'cooking (chopping)',
    '0510163840': 'cooking (stirring)',
    '0510163444': 'opening pill container',
    '0510163513': 'opening pill container',
    '0510163542': 'opening pill container',
    '0510164621': 'working on computer',
    '0511121410': 'still',
    '0511121542': 'talking on the phone',
    '0511124850': 'writing on whiteboard',
    '0511121954': 'drinking water',
    '0511130523': 'rinsing mouth with water',
    '0511130138': 'brushing teeth',
    '0511130920': 'wearing contact lenses',
    '0511131018': 'wearing contact lenses',
    '0511122214': 'talking on couch',
    '0511122813': 'relaxing on couch',
    '0511124349': 'cooking (chopping)',
    '0511124101': 'cooking (stirring)',
    '0511123142': 'opening pill container',
    '0511123218': 'opening pill container',
    '0511123238': 'opening pill container',
    '0511123806': 'working on computer',
    '0512172825': 'still',
    '0512171649': 'talking on the phone',
    '0512175502': 'writing on whiteboard',
    '0512173312': 'drinking water',
    '0512164800': 'rinsing mouth with water',
    '0512164529': 'brushing teeth',
    '0512165243': 'wearing contact lenses',
    '0512165327': 'wearing contact lenses',
    '0512174513': 'talking on couch',
    '0512174643': 'relaxing on couch',
    '0512171207': 'cooking (chopping)',
    '0512171444': 'cooking (stirring)',
    '0512173520': 'opening pill container',
    '0512173548': 'opening pill container',
    '0512173623': 'opening pill container',
    '0512170134': 'working on computer',
    '0512150222': 'still',
    '0512150451': 'talking on the phone',
    '0512154505': 'writing on whiteboard',
    '0512150912': 'drinking water',
    '0512155606': 'rinsing mouth with water',
    '0512155226': 'brushing teeth',
    '0512160143': 'wearing contact lenses',
    '0512160254': 'wearing contact lenses',
    '0512151230': 'talking on couch',
    '0512151444': 'relaxing on couch',
    '0512152943': 'cooking (chopping)',
    '0512152416': 'cooking (stirring)',
    '0512151857': 'opening pill container',
    '0512151934': 'opening pill container',
    '0512152013': 'opening pill container',
    '0512153758': 'working on computer'
}
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
