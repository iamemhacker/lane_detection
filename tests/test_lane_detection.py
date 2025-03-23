import cv2
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..', 'src')))
from lane_detection import detect  # noqa E402


def test_lane_detect():
    data_dir = on.getenv("TSET_DIR")
    if data_dir is None:
        print("Environment variable TSET_DIR is not set")
        assert False
    counter = 0
    gen = detect(TEST_FILE)
    while True:
        try:
            frame, lines = next(gen)
            line_image = np.copy(frame)
            for line in lines:
                cv2.line(line_image, line[:2], line[2:], (0, 255, 0), 3)
            counter += 1
            cv2.imwrite(f"/tmp/frame-lined{counter}.png", line_image)
            print(f"Frame {counter} written")
        except StopIteration:
            break
        except Exception:
            assert False

    assert counter > 0
