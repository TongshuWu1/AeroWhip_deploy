"""Replay metadata survives portable, weights-only checkpoint serialization."""
import io
import unittest

import numpy as np
import torch

from training.artifacts import checkpoint_replay, replay_from_checkpoint


class ReplayCheckpointMetadataTest(unittest.TestCase):
    def test_unicode_and_numeric_round_trip_weights_only(self):
        arrays = {
            'action_interface': np.array('acceleration_direct_v1'),
            'continuous_whip_fields': np.array(['fold_support', 'horizontal_reach_fraction']),
            'nested_labels': np.array([['world', 'relative'], ['outward', 'straightening']]),
            'empty_labels': np.empty(0, dtype='U8'),
            'time_s': np.array([0., .01, .02], dtype=np.float64),
            'state_valid': np.array([True, True, False]),
            'indices': np.array([1, 2, 3], dtype=np.int64),
        }
        saved = checkpoint_replay(arrays)
        self.assertIsInstance(saved['action_interface'], str)
        self.assertEqual(saved['continuous_whip_fields'], ['fold_support', 'horizontal_reach_fraction'])
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        restored = replay_from_checkpoint(torch.load(stream, weights_only=True))
        for name, original in arrays.items():
            np.testing.assert_array_equal(restored[name], original)
            self.assertEqual(restored[name].shape, original.shape)
            self.assertEqual(restored[name].dtype.kind, original.dtype.kind)

    def test_historical_tensor_only_replay(self):
        old = {'time_s': torch.tensor([0., .01], dtype=torch.float64),
               'state_valid': torch.tensor([True, False])}
        stream = io.BytesIO()
        torch.save(old, stream)
        stream.seek(0)
        restored = replay_from_checkpoint(torch.load(stream, weights_only=True))
        for name, value in old.items():
            np.testing.assert_array_equal(restored[name], value.numpy())
            self.assertEqual(restored[name].dtype, value.numpy().dtype)


if __name__ == '__main__':
    unittest.main()
