# Data samples

This directory will hold a small ncorev4 fixture (≤ 50 MB, 1 sequence) for
the README quickstart. Until that fixture is published, point
`run_inference.py --ncore-path` at your own ncorev4 dataset.

The HuggingFace mock at `instant_nurec/_hf_mock.py:get_sample_data_path()`
already references this directory by name (`ncorev4_sample/`) — when the
corp publishes the placeholder repo `nvidia/instant-nurec-kelvin`, the mock
will resolve the sample data here automatically.
