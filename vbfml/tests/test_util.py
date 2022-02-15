from unittest import TestCase

import numpy as np
import pandas as pd

from vbfml.util import MultiBatchBuffer


class TestMultiBatchBuffer(TestCase):
    def setUp(self):
        self.batch_size = 3
        self.buffer = MultiBatchBuffer(batch_size=self.batch_size)

        values = {}

        self.n_rows = 10
        values["a"] = np.array(range(self.n_rows))
        values["b"] = np.array(range(self.n_rows, 2 * self.n_rows))
        self.df = pd.DataFrame(values)

    def test_buffer_insertion(self):
        offset = 5
        self.buffer.set_multibatch(self.df, min_batch=offset)

        # Attempt to get non-existing batches
        for index in range(offset):
            self.assertFalse(index in self.buffer)
            with self.assertRaises(IndexError):
                self.buffer.get_batch_df(index)

        for index in range(offset, offset + len(self.df) // self.batch_size):
            self.assertTrue(index in self.buffer)
            df = self.buffer.get_batch_df(index)
            self.assertEqual(len(df.columns), len(self.df.columns))

    def test_batch_values(self):
        self.buffer.set_multibatch(self.df, min_batch=0)
        batch_df_0 = self.buffer.get_batch_df(0)
        self.assertEqual(len(batch_df_0), self.buffer.batch_size)
        self.assertListEqual(list(batch_df_0["a"]), list(range(self.buffer.batch_size)))

        batch_df_1 = self.buffer.get_batch_df(1)
        self.assertEqual(len(batch_df_1), self.buffer.batch_size)
        self.assertListEqual(
            list(batch_df_1["a"]),
            list(range(self.buffer.batch_size, 2 * self.buffer.batch_size)),
        )
