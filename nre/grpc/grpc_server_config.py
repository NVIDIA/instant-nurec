import warnings

from functools import wraps
from typing import Optional

import click

from nre.config.base_schema import BaseConfigSchema, Field
from nre.config.model import RendererBackend
from nre.utils.cli import config_to_cli_options


DEFAULT_DOWNLOAD_CACHE_DIR = "~/.cache/nre/downloaded_scenes"


class GrpcServerConfig(BaseConfigSchema):
    """Settings for the gRPC server."""

    artifact_glob: Optional[str] = Field(
        description="Glob expression to find artifacts. Must end in .usdz to find relevant files.", default=None
    )
    host: Optional[str] = Field(description="GRPC server host", default="localhost")
    port: int = Field(description="GRPC server port", default=8080)
    health_port: Optional[int] = Field(
        description="Port for gRPC health checks (grpc.health.v1.Health). If unset, health is served on the main gRPC server/port (legacy behavior). Set to a different port to run a dedicated health-only gRPC server.",
        default=None,
    )
    test_scenes_are_valid: bool = Field(
        description="Try to load each detected scene before coming online. Helps avoid crashes during long running jobs.",
        default=False,
    )
    renderer: RendererBackend = Field(
        description=(
            "Renderer backend: 'default' uses the artifact's trained renderer (PyTorch forward pass), "
            "'gsplat' forces GSplatRenderer, 'nrend' uses the fast NRendWrapper (direct C++/CUDA JIT)."
        ),
        default=RendererBackend.DEFAULT,
    )
    enable_difix: bool = Field(description="Use Difix/Harmonizer postprocessing.", default=False)
    # For Harmonizer postprocessing, use the following URL as --difix-url:
    # https://api.ngc.nvidia.com/v2/org/nvidia/team/nre/models/nurec-fixer/versions/cosmos_3dgut_fixer_harmonizer/files/harmonizer_nontemporal.pt
    difix_url: str = Field(
        description="URL of Difix/Harmonizer checkpoint.",
        default="https://api.ngc.nvidia.com/v2/org/nvidia/team/nre/models/nurec-fixer/versions/cosmos_3dgut/files/cosmos_3dgut.pt",
    )
    difix_cache: str = Field(description="Full path to local Difix/Harmonizer cache dir.", default="~/.cache/nre/difix")
    difix_model_filename: str = Field(description="Filename of Difix/Harmonizer checkpoint.", default="cosmos_3dgut.pt")
    difix_resolution: tuple[int, int] = Field(
        description="Resolution for Difix/Harmonizer processing.", default=(576, 1024)
    )
    ray_chunk_size: int = Field(
        description="Maximum number of rays processed in a single forward pass. Default: effectively unlimited. Set if seeing GPU OOMs.",
        default=2**62,
    )
    egocar_hood_dir: Optional[str] = Field(description="Directory with egocar hood images.", default=None)
    download_cache_dir: str = Field(
        description="Directory for downloaded scene files", default=DEFAULT_DOWNLOAD_CACHE_DIR
    )
    download_cache_size: int = Field(description="Maximum number of downloaded scenes to keep in cache", default=5)
    max_workers: int = Field(description="Maximum number of workers for the grpc server", default=1)
    enable_editing_actors: bool = Field(
        description="Enable editing the actors in a scene or use the original actor poses.", default=False
    )
    cache_size: int = Field(
        description="Maximum number of models to cache (count-based LRU). If OOM occurs during loading, spare backends are automatically evicted and load is retried.",
        default=10,
    )
    metrics_output_dir: Optional[str] = Field(
        description="Directory to save render time metrics. If not specified, metrics collection is disabled.",
        default=None,
    )


def grpc_cli_options():
    options = config_to_cli_options(GrpcServerConfig())

    # Deprecated flags kept for backward compatibility — ignored in favor of --renderer.
    deprecated_options = [
        click.option(
            "--enable-nrend/--no-enable-nrend",
            default=None,
            hidden=True,
            help="Deprecated: use --renderer instead.",
        ),
        click.option(
            "--use-gsplat/--no-use-gsplat",
            default=None,
            hidden=True,
            help="Deprecated: use --renderer instead.",
        ),
    ]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Redirect deprecated flags to --renderer, warn, then strip them.
            enable_nrend = kwargs.pop("enable_nrend", None)
            use_gsplat = kwargs.pop("use_gsplat", None)

            if enable_nrend is not None:
                warnings.warn(
                    "--enable-nrend/--no-enable-nrend is deprecated, use --renderer instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if enable_nrend and kwargs.get("renderer") == RendererBackend.DEFAULT:
                    kwargs["renderer"] = RendererBackend.NREND

            if use_gsplat is not None:
                warnings.warn(
                    "--use-gsplat/--no-use-gsplat is deprecated, use --renderer instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if use_gsplat and kwargs.get("renderer") == RendererBackend.DEFAULT:
                    kwargs["renderer"] = RendererBackend.GSPLAT

            return func(*args, **kwargs)

        wrapper.__click_params__ = list(getattr(func, "__click_params__", []))
        for option in reversed(deprecated_options + options):
            wrapper = option(wrapper)
        return wrapper

    return decorator
