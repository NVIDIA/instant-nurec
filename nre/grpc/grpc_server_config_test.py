# compare against the verbose definition of the old serve.py command
# and make sure the dynamically added options are the same

import json
import logging

from typing import Optional

import click

from click.testing import CliRunner

import nre.utils.cli as cli

from nre.config.version import get_version
from nre.grpc.grpc_server_config import DEFAULT_DOWNLOAD_CACHE_DIR, GrpcServerConfig, grpc_cli_options
from nre.grpc.serve import serve_grpc
from nre.utils.misc import unpack_optional


log = logging.getLogger("nre.grpc.grpc_server_config_test")


# deprecated call
@click.command("serve-grpc")
@click.option(
    "--artifact-glob",
    type=str,
    help="Glob expression to find artifacts. Must end in .usdz to find relevant files.",
    required=False,
    default=None,  # NOTE: this was added to the old command to make it compatible with the new command
)
@click.option(
    "--host",
    type=str,
    help="GRPC server host",
    default="localhost",
    required=False,
)
@click.option(
    "--port",
    type=int,
    help="GRPC server port",
    default=8080,  # NOTE: this was a string in the old command, fixed to int here
    required=False,
)
@click.option(
    "--health-port",
    type=int,
    help=(
        "Port for gRPC health checks (grpc.health.v1.Health). "
        "If unset, health is served on the main gRPC server/port (legacy behavior). "
        "Set to a different port to run a dedicated health-only gRPC server."
    ),
    default=None,
    required=False,
)
@click.option(
    "--test-scenes-are-valid/--no-test-scenes-are-valid",
    help="Try to load each detected scene before coming online. Helps avoid crashes during long running jobs.",
    default=False,
)
@click.option(
    "--renderer",
    type=click.Choice(["default", "gsplat", "nrend"], case_sensitive=False),
    help=(
        "Renderer backend: 'default' uses the artifact's trained renderer (PyTorch forward pass), "
        "'gsplat' forces GSplatRenderer, 'nrend' uses the fast NRendWrapper (direct C++/CUDA JIT)."
    ),
    default="default",
)
@click.option(
    "--enable-difix/--no-enable-difix",  # NOTE: added --no- prefix to be consistent with other flag declarations
    is_flag=True,
    help="Use Difix/Harmonizer postprocessing.",
    default=False,
)
@click.option(
    "--difix-url",
    type=str,
    help="URL of Difix/Harmonizer checkpoint.",
    default="https://api.ngc.nvidia.com/v2/org/nvidia/team/nre/models/nurec-fixer/versions/cosmos_3dgut/files/cosmos_3dgut.pt",
)
@click.option(
    "--difix-cache",
    type=str,
    help="Full path to local Difix/Harmonizer cache dir.",
    default="~/.cache/nre/difix",
)
@click.option(
    "--difix-model-filename",
    type=str,
    help="Filename of Difix/Harmonizer checkpoint.",
    default="cosmos_3dgut.pt",
)
@click.option(
    "--difix-resolution",
    type=click.Tuple([int, int]),
    help="Resolution for Difix/Harmonizer processing.",
    default=(576, 1024),
)
@cli.scopedtimer_cli_options(print_func=log.info)
@click.option(
    "--ray-chunk-size",
    type=int,
    help="Maximum number of rays processed in a single forward pass. Default: effectively unlimited. Set if seeing GPU OOMs.",
    default=2**62,
)
@click.option(
    "--egocar-hood-dir",
    type=str,
    help="Directory with egocar hood images.",
    default=None,
)
@click.option(
    "--download-cache-dir",
    type=str,
    help="Directory for downloaded scene files",
    default=DEFAULT_DOWNLOAD_CACHE_DIR,
)
@click.option(
    "--download-cache-size",
    type=int,
    help="Maximum number of downloaded scenes to keep in cache",
    default=5,
)
@click.option(
    "--max-workers",
    type=int,
    help="Maximum number of workers for the grpc server",
    default=1,
)
@click.option(
    "--metrics-output-dir",
    type=str,
    help="Directory to save render time metrics. If not specified, metrics collection is disabled.",
    default=None,
)
@click.option(
    "--enable-editing-actors/--no-enable-editing-actors",  # NOTE: added --no- prefix to be consistent with other flag declarations
    is_flag=True,
    help="Enable editing the actors in a scene or use the original actor poses.",
    default=False,
)
@click.option(
    "--cache-size",
    type=int,
    help="Maximum number of models to cache (count-based LRU). If OOM occurs during loading, spare backends are automatically evicted and load is retried.",
    default=10,
)
@click.option(
    "--enable-nrend/--no-enable-nrend",
    default=None,
    hidden=True,
    help="Deprecated: use --renderer instead.",
)
@click.option(
    "--use-gsplat/--no-use-gsplat",
    default=None,
    hidden=True,
    help="Deprecated: use --renderer instead.",
)
@click.version_option(version=str(unpack_optional(get_version(), default="version-not-available")))
def serve_grpc_old(
    artifact_glob: Optional[str],
    host: str,
    port: int,
    health_port: Optional[int],
    test_scenes_are_valid: bool,
    renderer: str,
    enable_nrend: Optional[bool],
    use_gsplat: Optional[bool],
    enable_difix: bool,
    difix_url: str,
    difix_cache: str,
    difix_model_filename: str,
    difix_resolution: tuple[int, int],
    enable_timing: bool,
    timing_verbosity: str,
    timing_logfile: Optional[str],
    timing_synchronize: bool,
    profiling_backend: str,
    ray_chunk_size: int,
    egocar_hood_dir: Optional[str],
    download_cache_dir: str,
    download_cache_size: int,
    max_workers: int,
    cache_size: int,
    metrics_output_dir: Optional[str],
    enable_editing_actors: Optional[bool],
) -> click.Context:
    """Neural Rendering gRPC server"""
    return click.get_current_context()


def test_dynamic_grpc_server_command_equals_old_command():
    dynamic_command = serve_grpc
    old_command = serve_grpc_old

    assert isinstance(dynamic_command, click.Command)
    assert isinstance(old_command, click.Command)

    # also assert the help messages are the same
    assert dynamic_command.help == old_command.help
    assert dynamic_command.name == old_command.name

    # first sort params by name
    dynamic_command_params_sorted = sorted(dynamic_command.params, key=lambda x: str(x.name))
    old_command_params_sorted = sorted(old_command.params, key=lambda x: str(x.name))

    for dynamic_param, old_param in zip(dynamic_command_params_sorted, old_command_params_sorted, strict=True):
        print(f"# dynamic_param.name: {dynamic_param.name} - old_param.name: {old_param.name}")
        assert dynamic_param.name == old_param.name
        assert dynamic_param.default == old_param.default
        print(f"# dynamic_param.type: {type(dynamic_param.type)} - old_param.type: {type(old_param.type)}")
        assert type(dynamic_param.type) == type(old_param.type)
        assert dynamic_param.required == old_param.required
        assert dynamic_param.get_help_record(click.Context(dynamic_command)) == old_param.get_help_record(
            click.Context(old_command)
        )


@click.command("serve-grpc-test")
@grpc_cli_options()
def serve_grpc_test_command(**kwargs):  # pragma: no cover - invoked by CliRunner
    # Instantiate config to trigger validation
    config = GrpcServerConfig(**kwargs)
    click.echo(json.dumps(config.model_dump(mode="json"), sort_keys=True))


def _invoke_grpc_test_command(args: list[str]) -> dict:
    runner = CliRunner()
    result = runner.invoke(serve_grpc_test_command, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output.strip())


def test_renderer_default_is_default():
    params = _invoke_grpc_test_command([])
    assert params["renderer"] == "default"


def test_renderer_gsplat():
    params = _invoke_grpc_test_command(["--renderer", "gsplat"])
    assert params["renderer"] == "gsplat"


def test_renderer_nrend():
    params = _invoke_grpc_test_command(["--renderer", "nrend"])
    assert params["renderer"] == "nrend"


def test_deprecated_enable_nrend_redirects_to_renderer():
    params = _invoke_grpc_test_command(["--enable-nrend"])
    assert params["renderer"] == "nrend"


def test_deprecated_use_gsplat_redirects_to_renderer():
    params = _invoke_grpc_test_command(["--use-gsplat"])
    assert params["renderer"] == "gsplat"


# --no-enable-nrend, just like the "default" --renderer arg, will
# redirect to the proper renderer based on which one was used during training
def test_deprecated_no_enable_nrend_keeps_default():
    params = _invoke_grpc_test_command(["--no-enable-nrend"])
    assert params["renderer"] == "default"


# Make sure that --renderer arg takes precedence over deprecated flags
def test_explicit_renderer_overrides_deprecated_flag():
    params = _invoke_grpc_test_command(["--renderer", "gsplat", "--enable-nrend"])
    assert params["renderer"] == "gsplat"
