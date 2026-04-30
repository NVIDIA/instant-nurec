# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import json
import logging

from pathlib import Path

import torch.utils.data
import websockets.sync.client as websockets_client

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection

from nre.nrm.config.dataset import WebSocketNCoreNRMDatasetConfig
from nre.nrm.datasets.nrm_ncore import NCoreNRMDataset
from nre.nrm.datasets.registry import register as register_dataset
from nre.utils.batch import NRMDataBatch


logger = logging.getLogger(__name__)


@register_dataset("nrm-websocket-ncore")
class WebSocketNCoreNRMDataset(NCoreNRMDataset):
    """
    NCore dataset that listens to WebSocket requests for sequence selection.
    Inherits all data loading logic from NCoreNRMDataset, but remaps the index
    based on WebSocket messages that specify which sequence to load.

    Must be run in the main process (num_workers=0) since WebSocket connections cannot be shared.
    """

    MAX_BATCHES = 100000

    def __init__(self, config: WebSocketNCoreNRMDatasetConfig, split: str = "train"):
        super().__init__(self._optimize_config_for_streaming(config), split)
        self.ws_config = config
        self.socket: ClientConnection | None = None

        # Current sequence and the index within the sequence that is being loaded.
        self.current_local_batch_idx: int = 0
        self.current_sequence: str | None = None

        # Build a quick mapping from sequence name to index for fast lookup.
        self.sequence_mapping: dict[str, int] = {}
        for idx, sequence in enumerate(self.ncore_json_paths):
            path = str(sequence.stem)
            self.sequence_mapping[path] = idx

    @staticmethod
    def _optimize_config_for_streaming(config: WebSocketNCoreNRMDatasetConfig) -> WebSocketNCoreNRMDatasetConfig:
        """Optimize the config for streaming from S3."""
        config = config.model_copy(deep=True)
        config.cache_loaders_and_sensors = True
        config.s3_block_size_mb = 64
        config.s3_cache_type = "blockcache"
        return config

    def _connect(self) -> None:
        """Connect to WebSocket server."""
        url = f"ws://{self.ws_config.ws_server_addr}:{self.ws_config.ws_server_port}"
        logger.info(f"Connecting to WebSocket server at {url}...")
        self.socket = websockets_client.connect(url, ping_timeout=None)
        logger.info(f"Connected to WebSocket server at {url}")

    def _recv(self) -> str:
        """Receive message from WebSocket, reconnecting if connection was closed."""
        assert self.socket is not None
        while True:
            try:
                msg = self.socket.recv()
                return msg if isinstance(msg, str) else msg.decode("utf-8")
            except ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                self._connect()
                continue

    def __len__(self) -> int:
        """Returns a large number to allow continuous iteration."""
        return self.MAX_BATCHES

    def __getitem__(self, batch_idx: int) -> NRMDataBatch:
        # Lazily connect to WebSocket server on first access
        if self.socket is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                raise RuntimeError(
                    f"WebSocketNCoreNRMDataset must run in main process (num_workers=0), "
                    f"but running in worker {worker_info.id}"
                )
            self._connect()

        # Block for next sequence when starting or after exhausting current sequence
        if self.current_local_batch_idx == 0:
            sequence_name = None
            while sequence_name is None:
                message = self._recv()
                sequence = self._parse_sequence(message)
                if sequence is None:
                    continue
                sequence_name = Path(sequence).stem
                if sequence_name not in self.sequence_mapping:
                    logger.warning(f"Sequence {sequence_name} not found in sequence_mapping, skipping...")
                    sequence_name = None
            self.current_sequence = sequence_name

        # Remap index: compute global index into parent dataset (current_sequence set above or in prior call)
        assert self.current_sequence is not None
        local_batch_idx = self.current_local_batch_idx
        logger.info(f"Getting batch {local_batch_idx} from sequence {self.current_sequence}")
        global_idx = self.sequence_mapping[self.current_sequence] * self.num_samples_per_sequence + local_batch_idx

        # Use parent's __getitem__ to get the actual data
        data_batch = super().__getitem__(global_idx)

        self.current_local_batch_idx += 1
        if self.current_local_batch_idx >= self.num_samples_per_sequence:
            self.current_local_batch_idx = 0

        return data_batch

    def _parse_sequence(self, message: str) -> str | None:
        """Parse a WebSocket message to extract the sequence name.

        Expected input: a JSON text message of the form
            {"type": "broadcast", "data": "<sequence_path_or_name>"}
        where "data" is the sequence to load (e.g. a path like "path/to/clip/clip.json"
        or a stem like "clipgt-xxx"). The stem is used to look up the sequence in
        sequence_mapping. Any other message (non-JSON, or type != "broadcast") is
        ignored and returns None, so the receiver will keep waiting for a valid
        broadcast.
        """
        try:
            data = json.loads(message)
            method = data.get("type", "unknown")
            if method == "broadcast":
                logger.info(f"RECEIVED: {data.get('data')}")
                sequence = data.get("data")
                return sequence

        except json.JSONDecodeError:
            pass

        logger.info(f"RECEIVED (raw): {message}")
        return None
