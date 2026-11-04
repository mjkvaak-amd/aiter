# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL QSA device kernels.

  ``sparse_decode_attn.py``    one CTA per (row, kv head, KV split), reading the
                               selected KV straight from the token pool
  ``sparse_decode_reduce.py``  merges the KV-split partials
  ``indexer_select.py``, ``sparse_gqa.py``  the K1 indexer select and K2 sparse GQA

The public entry points are ``aiter.ops.flydsl.qsa_sparse_decode`` and
``aiter.ops.flydsl.qsa``. The fp32 reference lives in ``op_tests/qsa_ref.py``.
"""
