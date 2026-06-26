# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: ep_tune_lookup_topk routed-topk normalization (no GPU, fast).

Covers the tuned-FMoE-config lookup topk used under expert parallelism:
  - non-EP / missing inputs -> runtime topk unchanged
  - DSv3 / SGLang fake-expert column convention -> routed_topk (topk - 1)
  - vLLM / GLM (no fake column) -> runtime topk unchanged

Run:
    python3 -m unittest op_tests.tuning_tests.test_ep_lookup_topk -v
"""

import unittest

try:
    import torch

    from aiter.fused_moe import ep_tune_lookup_topk

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    _IMPORT_ERROR = exc


@unittest.skipUnless(_IMPORT_ERROR is None, f"import unavailable: {_IMPORT_ERROR}")
class TestEpTuneLookupTopk(unittest.TestCase):
    """Unit tests for ``ep_tune_lookup_topk`` (pure CPU tensor logic)."""

    @staticmethod
    def _topk_ids(rows, width, last_col, routed_ids=0):
        """Build a ``[rows, width]`` topk_ids tensor.

        The leading ``width - 1`` (routed) columns are filled with
        ``routed_ids``; ``last_col`` sets the final column, either as an int
        broadcast to every row or as a per-row list.
        """
        if isinstance(last_col, int):
            last = [last_col] * rows
        else:
            last = list(last_col)
        return torch.tensor(
            [[routed_ids] * (width - 1) + [last[i]] for i in range(rows)],
            dtype=torch.int32,
        )

    @staticmethod
    def _expert_mask(numel, fake_masked=True):
        """expert_mask of length ``numel``; the fake (last) slot is masked."""
        mask = torch.ones(numel, dtype=torch.int32)
        mask[-1] = 0 if fake_masked else 1
        return mask

    def test_non_ep_returns_topk(self):
        # No expert_mask -> not an EP call; topk untouched.
        ids = self._topk_ids(4, width=8, last_col=7)
        self.assertEqual(ep_tune_lookup_topk(8, expert_mask=None, topk_ids=ids), 8)

    def test_no_topk_ids_returns_topk(self):
        mask = self._expert_mask(9)
        self.assertEqual(ep_tune_lookup_topk(8, expert_mask=mask, topk_ids=None), 8)

    def test_topk_le_one_returns_topk(self):
        # Can't strip below 1 expert; guard against degenerate topk.
        mask = self._expert_mask(9)
        ids = self._topk_ids(4, width=1, last_col=8)
        self.assertEqual(ep_tune_lookup_topk(1, expert_mask=mask, topk_ids=ids), 1)
        self.assertEqual(ep_tune_lookup_topk(0, expert_mask=mask, topk_ids=ids), 0)

    def test_empty_topk_ids_returns_topk(self):
        # Zero-token batch: nothing to inspect, keep runtime topk.
        mask = self._expert_mask(9)
        empty = torch.empty((0, 8), dtype=torch.int32)
        self.assertEqual(ep_tune_lookup_topk(8, expert_mask=mask, topk_ids=empty), 8)

    def test_dsv3_fake_column_stripped(self):
        # DSv3/SGLang: last column is the always-masked fake expert id.
        # runtime topk = routed_topk + 1 -> lookup should use routed_topk.
        mask = self._expert_mask(9, fake_masked=True)  # fake_id = 8, masked
        fake_id = mask.numel() - 1
        ids = self._topk_ids(6, width=9, last_col=fake_id, routed_ids=3)
        self.assertEqual(ep_tune_lookup_topk(9, expert_mask=mask, topk_ids=ids), 8)

    def test_vllm_no_fake_column(self):
        # vLLM/GLM: topk_ids width already equals routed topk; last column is a
        # real routed expert, not the fake id -> topk unchanged (GLM topk=8).
        mask = self._expert_mask(9, fake_masked=True)
        ids = self._topk_ids(6, width=8, last_col=7, routed_ids=2)  # 7 != fake (8)
        self.assertEqual(ep_tune_lookup_topk(8, expert_mask=mask, topk_ids=ids), 8)

    def test_fake_expert_not_masked_returns_topk(self):
        # If the candidate fake id is NOT masked it is a real expert; don't strip
        # even though the last column uniformly points at it.
        mask = self._expert_mask(9, fake_masked=False)  # last slot active
        fake_id = mask.numel() - 1
        ids = self._topk_ids(6, width=9, last_col=fake_id, routed_ids=1)
        self.assertEqual(ep_tune_lookup_topk(9, expert_mask=mask, topk_ids=ids), 9)

    def test_fake_column_not_uniform_returns_topk(self):
        # Last column must be the fake id for ALL tokens to count as a fake
        # column; a single deviating row keeps the runtime topk.
        mask = self._expert_mask(9, fake_masked=True)
        fake_id = mask.numel() - 1
        last = [fake_id] * 6
        last[2] = 4  # one token routes its last slot to a real expert
        ids = self._topk_ids(6, width=9, last_col=last, routed_ids=0)
        self.assertEqual(ep_tune_lookup_topk(9, expert_mask=mask, topk_ids=ids), 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
