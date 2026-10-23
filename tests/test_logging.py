import io
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from wmlib.utils.logger import TensorBoardOutputPytorch, encode_gif


class VideoLoggingTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_video_encodes_as_readable_gif(self):
        frames = np.zeros((6, 16, 16, 3), dtype=np.uint8)
        for index in range(len(frames)):
            frames[index, :, :, 0] = index * 40
        encoded = encode_gif(frames, 20)
        with Image.open(io.BytesIO(encoded)) as image:
            self.assertEqual(image.format, "GIF")
            self.assertEqual(image.size, (16, 16))
            self.assertEqual(image.n_frames, len(frames))

    def test_encoding_failure_logs_first_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            output = TensorBoardOutputPytorch(directory)
            video = np.zeros((6, 16, 16, 3), dtype=np.uint8)
            with mock.patch(
                "wmlib.utils.logger.encode_gif",
                side_effect=OSError("encoder unavailable"),
            ):
                output([(10, "policy", video)])
            output._writer.close()
            events = EventAccumulator(directory).Reload()
            saved = events.Images("policy")
            self.assertEqual(len(saved), 1)
            self.assertEqual(
                (saved[0].width, saved[0].height, saved[0].step), (16, 16, 10)
            )


if __name__ == "__main__":
    unittest.main()
