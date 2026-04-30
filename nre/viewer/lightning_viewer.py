# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from threading import Lock
from time import sleep
from typing import Any, Callable, Dict, Mapping, Optional

import torch

from pytorch_lightning import Callback, LightningModule, Trainer

from nre.config.viewer import ViewerConfig
from nre.datasets.ncore import NCOREDataSource
from nre.models.gaussians.gaussians_composite import GaussiansComposite
from nre.render import RenderableModel
from nre.systems.base import BaseSystem
from nre.systems.gaussians import GaussiansSystem
from nre.utils.batch import NRMDataBatch
from nre.utils.misc import decorate_all, rank_zero_only
from nre.viewer.abstract_viewer import AbstractViewer
from nre.viewer.dataset_interface import ViewerDatasetInterface
from nre.viewer.lock import LockLike
from nre.viewer.ncore_dataset_interface import ViewerNCOREInterface
from nre.viewer.nrm_dataset_interface import (
    EmptyDatasetInterface,
    ViewerNRMInterface,
    get_nrm_primitive_from_loss_return,
    postprocess_nrm_batch_and_primitive,
)


log = logging.getLogger(__name__)

UpdateSceneFnType = Callable[[RenderableModel, ViewerDatasetInterface, bool], None]


def is_nrm_system(system: Any) -> bool:
    """
    Check if the given system is an instance of BaseNRMSystem or any of its subclasses.
    This is mainly used to bypass importing BaseNRMSystem (and hence all NRM-related components) for the pycena build.
    """
    return any(cls.__name__ == "BaseNRMSystem" for cls in system.__class__.mro())


class ObserverCallback(Callback):
    """
    A pytorch_lightning callback which takes care of two things important for training observers like the viewer server:
        1) locking/unlocking the system on each train/val/test/predict step to avoid overlap with, e.g., viewer server's renders
        2) keeps tracks if any exceptions happened during training and potentially keeps an observer (like a viewer) online after
           the Trainer.fit loop
    """

    def __init__(
        self,
        system_lock: LockLike,
        start_server_fn: Callable[[], None],
        update_scene_fn: UpdateSceneFnType | None,
        remain_after_trainer_loop: bool,
    ):
        super().__init__()

        self.system_lock = system_lock
        self.start_server_fn = start_server_fn
        self.update_scene_fn = update_scene_fn
        self.remain_after_trainer_loop = remain_after_trainer_loop

        # pytorch_lightning "graceful shutdown" functionality catches exceptions so we keep a flag
        # instead of relying on being interrupted by the exception propagating
        self.trainer_raised: bool = False

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        self.trainer_raised = True

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_server_fn()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.maybe_remain(pl_module=pl_module)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_server_fn()

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.maybe_remain(pl_module=pl_module)

    def maybe_remain(self, pl_module: LightningModule) -> None:
        if not self.remain_after_trainer_loop or self.trainer_raised:
            return

        log.info("Execution finished, blocking to allow viewing. Send a keyboard interrupt to stop.")

        # once trainer is done, the system goes back to CPU - bring it back
        # TODO: maybe we should be more specific _which_ cuda should be used?
        pl_module.cuda()

        # endless loop
        while True:
            sleep(1.0)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_local_idx: int
    ) -> None:
        self.system_lock.acquire()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_local_idx: int,
    ) -> None:
        self.system_lock.release()
        if self.update_scene_fn is None:
            return
        if is_nrm_system(pl_module) and (loss_return := pl_module.last_loss_return) is not None:
            if (nrm_primitive := get_nrm_primitive_from_loss_return(loss_return)) is not None:
                batch, nrm_primitive = postprocess_nrm_batch_and_primitive(batch, nrm_primitive)
                self.update_scene_fn(RenderableModel(nrm_primitive.detach()), ViewerNRMInterface(batch), False)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_local_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.system_lock.acquire()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_local_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.system_lock.release()
        if self.update_scene_fn is None:
            return
        if is_nrm_system(pl_module) and isinstance(outputs, dict):
            if (nrm_primitive := get_nrm_primitive_from_loss_return(outputs["result"])) is not None:
                batch, nrm_primitive = postprocess_nrm_batch_and_primitive(batch, nrm_primitive)
                self.update_scene_fn(RenderableModel(nrm_primitive), ViewerNRMInterface(batch), False)

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_local_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.system_lock.acquire()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_local_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.system_lock.release()
        if self.update_scene_fn is None:
            return
        if is_nrm_system(pl_module) and isinstance(outputs, dict):
            if (nrm_primitive := get_nrm_primitive_from_loss_return(outputs["result"])) is not None:
                batch, nrm_primitive = postprocess_nrm_batch_and_primitive(batch, nrm_primitive)
                self.update_scene_fn(RenderableModel(nrm_primitive), ViewerNRMInterface(batch), True)

    def on_predict_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_local_idx: int, dataloader_idx: int = 0
    ) -> None:
        self.system_lock.acquire()

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_local_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.system_lock.release()
        if self.update_scene_fn is None:
            return
        if (
            is_nrm_system(pl_module)
            and isinstance(outputs, dict)
            and (primitives := outputs.get("primitives")) is not None
        ):
            batch = outputs.get("batch", batch)
            assert len(primitives) == 1, "Predict step should return a single primitive with batch size 1"
            assert len(batch) == len(primitives), "Length of batch and primitives should match"
            assert isinstance(batch, NRMDataBatch)
            batch, nrm_primitive = postprocess_nrm_batch_and_primitive(batch, primitives[0])
            self.update_scene_fn(RenderableModel(nrm_primitive), ViewerNRMInterface(batch), True)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_server_fn()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.maybe_remain(pl_module=pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_server_fn()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.maybe_remain(pl_module=pl_module)


@decorate_all([rank_zero_only])
class LightningSOViewer(AbstractViewer):
    """
    A viewer, to be created via Viewer.create and used as a context manager over a running
    pytorch_lightning trainer loop.
    """

    def __init__(
        self,
        model: GaussiansComposite,
        datasource: NCOREDataSource,
        config: ViewerConfig,
    ) -> None:
        """
        Internal, use maybe_create_lightning_viewer instead
        """
        super().__init__()

        self.config = config
        self.model = RenderableModel(model)
        self.host = config.host
        self.port = config.port
        self._datasource_interface: Optional[ViewerNCOREInterface] = None
        self.datasource = datasource
        self.system_lock = Lock()

    def get_dataset_interface(self) -> ViewerNCOREInterface:
        if self._datasource_interface is None:
            self._datasource_interface = ViewerNCOREInterface(self.datasource)
        return self._datasource_interface


@decorate_all([rank_zero_only])
class LightningNRMViewer(AbstractViewer):
    """
    For NRM systems, we reconstruct multiple scenes through a training/testing session, so we need to
    have an 'update scene' functionality to allow the user to change the current scene being visualized.

    For training, the logic is:
    - NRM is training in the background, iterating through scenes (batches).
    - The viewer picks up the first set of primitives and allows for interactive viewing.
    - At any point the user can click the "next scene" button and switch to the latest scene (batch).
    - The user doesn't have explicit control over which scene they are looking at.

    For testing, the logic is the same as training, but the test progress will block until the user clicks the "next scene" button.
    """

    def __init__(self, config: ViewerConfig) -> None:
        """
        Internal, use maybe_create_lightning_viewer instead
        """
        super().__init__()

        self.config = config
        self.model = None
        self.host = config.host
        self.port = config.port
        self._datasource_interface: Optional[ViewerNRMInterface] = None
        self.system_lock = Lock()

    def get_dataset_interface(self) -> ViewerDatasetInterface:
        if self._datasource_interface is None:
            return EmptyDatasetInterface()
        return self._datasource_interface

    def supports_clear_current_scene(self) -> bool:
        return True

    def clear_current_scene(self) -> None:
        with self.scene_lock:
            self.model = None
            self._datasource_interface = None

    def update_scene(
        self,
        renderable_model: RenderableModel,
        dataset_interface: ViewerDatasetInterface,
        block_until_empty_again: bool = False,
    ):
        """
        Sets a new scene in the viewer if not already set. Scene is unset at the beginning of viewer or after the user
        clears it with a button press.
        This method is called from training loop callbacks like `on_{train,val,test}_batch_end` since each forward pass
        provides a new scene that may be viewed.
        `block_until_empty_again` makes the update non-optional, blocking until user clears the previous scene instead of skipping.
        """
        # Update the scene
        with self.scene_lock:
            # If block_until_empty_again=True mostly likely both below are None
            if self.model is None and self._datasource_interface is None:
                self.model = renderable_model
                assert isinstance(dataset_interface, ViewerNRMInterface)
                self._datasource_interface = dataset_interface

        # Block the execution before the scene is empty again (via user clicks the clear scene button)
        if block_until_empty_again and not self.config.force_update_scene:
            while True:
                with self.scene_lock:
                    if self.model is None and self._datasource_interface is None:
                        break
                sleep(0.1)

    def get_current_scene_id(self) -> tuple[int, int]:
        if self.model is None or self._datasource_interface is None:
            return -1, -1
        return super().get_current_scene_id()


def maybe_create_lightning_viewer(system: BaseSystem, config: ViewerConfig) -> ObserverCallback | None:
    """
    A factory function which creates and returns an ObserverCallback wrapping a viewer if appropriate or None otherwise.

    Args:
        - system: the system to use for rendering
        - config: viewer configuration dict
    """
    if not config.enabled:
        return None

    viewer: AbstractViewer
    if isinstance(system, (GaussiansSystem)):
        datasource = system.datamodule.get_datasource()
        if not isinstance(datasource, NCOREDataSource):
            log.warning(
                f"Skipping viewer (enabled in config): SO viewer only supports NCOREDataSource (found {type(datasource)=})"
            )
            return None

        viewer = LightningSOViewer(system.model, datasource, config)

    elif is_nrm_system(system):
        viewer = LightningNRMViewer(config)

    else:
        log.warning(f"Skipping viewer (enabled in config): {type(system)=} is not supported.")
        return None

    return ObserverCallback(
        system_lock=viewer.system_lock,
        start_server_fn=viewer.start_server,
        update_scene_fn=viewer.update_scene,
        remain_after_trainer_loop=config.remain_after_trainer_loop,
    )
