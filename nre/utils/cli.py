# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""CLI utilities for NRE commands."""

from __future__ import annotations

import json
import logging

from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast, get_args, get_origin

import click

from nre.config.base_schema import BaseConfigSchema, Field
from nre.config.model import RendererBackend
from nre.config.scopedtimer import ProfilerBackend, VerbosityLevel, VerbosityLiteral
from nre.config.version import get_version
from nre.utils.profiling import configure_scopedtimer_from_cli


log = logging.getLogger(__name__)


class ScopedTimerConfig(BaseConfigSchema):
    enable_timing: bool = Field(
        default=False, description="Enable timing of the different parts of the rendering pipeline."
    )
    timing_verbosity: VerbosityLiteral = Field(
        default=VerbosityLevel.BASIC, description="Verbosity level for timing output."
    )
    timing_logfile: Optional[str] = Field(
        default=None, description="File to write timing results to (e.g., timing.log)."
    )
    timing_synchronize: bool = Field(default=False, description="Synchronize GPU before timing measurements.")
    profiling_backend: ProfilerBackend = Field(
        default=ProfilerBackend.NONE, description="Profiler backend to use: NONE (default), TRACY, or NVTX."
    )


def config_to_cli_options(config: BaseConfigSchema) -> list:
    # Dynamically add options based on the config fields
    options = []
    for field_name, field_info in config.model_fields.items():
        field_name_dashed = field_name.replace("_", "-")
        option_name = f"--{field_name_dashed}"
        field_type: Any = field_info.annotation
        default_value = field_info.default
        description = field_info.description

        callback: Any = None

        # Handle boolean flags
        is_flag = False
        param_declarations = option_name

        def is_optional(tp) -> bool:
            origin = get_origin(tp)
            if origin is Union:
                args = get_args(tp)
                return type(None) in args

            return False

        if is_optional(field_type):

            def get_optional_type(tp):
                """Helper function to get the type of the optional type.

                Note that the optional type is restricted to Optional[T] for some T.
                So no deeper nested structures are allowed.
                """
                if len(get_args(tp)) != 2:
                    raise ValueError(f"Unsupported Union type: {tp}. Only Optional[T] is supported.")

                return get_args(tp)[0]

            field_type = get_optional_type(field_type)
        if field_type == bool:
            param_declarations = f"{option_name}/--no-{field_name_dashed}"
            is_flag = True

        if field_type == VerbosityLiteral:
            field_type = click.Choice(get_args(field_type), case_sensitive=False)
            callback = lambda _, __, value: cast(VerbosityLiteral, value.upper())
        elif field_type == ProfilerBackend:
            default_value = str(default_value.value)
            callback = lambda _, __, value: ProfilerBackend[value.upper()]
            field_type = click.Choice([str(enum_value.value) for enum_value in ProfilerBackend], case_sensitive=False)
        elif field_type == RendererBackend:
            default_value = str(default_value.value)
            callback = lambda _, __, value: RendererBackend(value.lower())
            field_type = click.Choice([str(enum_value.value) for enum_value in RendererBackend], case_sensitive=False)
        elif get_origin(field_type) == tuple:
            field_type = click.Tuple(get_args(field_type))

        option = click.option(
            param_declarations,
            is_flag=is_flag,
            default=default_value,
            type=field_type,
            help=description,
            callback=callback,
        )
        options.append(option)
    return options


def scopedtimer_cli_options(print_func: Optional[Callable[[str], None]] = None):
    """Shared CLI options for ScopedTimer configuration."""
    options = [
        click.option(
            "--enable-timing",
            is_flag=True,
            help="Enable timing of the different parts of the rendering pipeline.",
            default=False,
        ),
        click.option(
            "--timing-verbosity",
            type=click.Choice(["NONE", "SUMMARY", "BASIC", "DETAILS"], case_sensitive=False),
            help="Verbosity level for timing output.",
            default="BASIC",
            callback=lambda _, __, value: cast(VerbosityLiteral, value.upper()),
        ),
        click.option(
            "--timing-logfile",
            type=str,
            help="File to write timing results to (e.g., timing.log).",
            default=None,
        ),
        click.option(
            "--timing-synchronize",
            is_flag=True,
            help="Synchronize GPU before timing measurements.",
            default=False,
        ),
        click.option(
            "--profiling-backend",
            type=click.Choice(["NONE", "TRACY", "NVTX"], case_sensitive=False),
            help="Profiling backend to use: NONE (default), TRACY, or NVTX.",
            default="NONE",
            callback=lambda _, __, value: ProfilerBackend[value.upper()],
        ),
    ]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            enable_timing = kwargs.pop("enable_timing", False)
            timing_verbosity = cast(VerbosityLiteral, kwargs.pop("timing_verbosity", "BASIC"))
            timing_logfile = kwargs.pop("timing_logfile", None)
            timing_synchronize = kwargs.pop("timing_synchronize", False)
            profiling_backend = kwargs.pop("profiling_backend", ProfilerBackend.NONE)
            configure_scopedtimer_from_cli(
                enable_timing=enable_timing,
                timing_verbosity=timing_verbosity,
                timing_logfile=timing_logfile,
                timing_synchronize=timing_synchronize,
                profiling_backend=profiling_backend,
                print_func=print_func,
            )
            return func(*args, **kwargs)

        # Copy click parameters from the original function
        setattr(wrapper, "__click_params__", list(getattr(func, "__click_params__", [])))
        for option in reversed(options):
            wrapper = option(wrapper)
        return wrapper

    return decorator


class SettingsCollector:
    """Captures Click CLI parameters and exports them for traceability."""

    def __init__(
        self,
        command_name: str,
        args: dict[str, Any],
        timestamp: datetime,
        nre_version: str,
    ) -> None:
        self.command_name = command_name
        self.args = args
        self.timestamp = timestamp
        self.nre_version = nre_version

    @classmethod
    def from_click_context(cls, ctx: click.Context, command_name: str) -> SettingsCollector:
        """Create a SettingsCollector from a Click context.

        Args:
            ctx: The Click context containing CLI parameters.
            command_name: Name of the command being executed.

        Returns:
            A SettingsCollector instance with captured settings.
        """
        version = get_version()
        nre_version = version.version_string if version else "unknown"

        args = cls._serialize_params(ctx.params)

        return cls(
            command_name=command_name,
            args=args,
            timestamp=datetime.now(timezone.utc),
            nre_version=nre_version,
        )

    @staticmethod
    def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
        """Serialize Click parameters to JSON-compatible types.

        Handles:
        - Path -> str
        - tuple -> list
        - Other types pass through (str, int, float, bool, None)
        """
        result = {}
        for key, value in params.items():
            result[key] = SettingsCollector._serialize_value(value)
        return result

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a single value to a JSON-compatible type."""
        if isinstance(value, Path):
            return str(value)
        elif isinstance(value, Enum):
            return value.name
        elif isinstance(value, tuple) or isinstance(value, list):
            return [SettingsCollector._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: SettingsCollector._serialize_value(v) for k, v in value.items()}
        return value

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary representation."""
        return {
            "nre_version": self.nre_version,
            "timestamp": self.timestamp.isoformat(),
            "command": self.command_name,
            "args": self.args,
        }

    def log_settings(self, logger: logging.Logger) -> None:
        """Log the settings using the provided logger."""
        args_json = json.dumps(self.args, indent=2)
        logger.info(f"{self.command_name} args (nre_version={self.nre_version}):\n{args_json}")

    def save_json(self, output_path: Path) -> Path:
        """Save settings to a JSON file.

        Args:
            output_path: Full path to the output JSON file.

        Returns:
            Path to the saved JSON file.
        """
        output_path = Path(output_path)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        log.info(f"Saved settings to {output_path}")
        return output_path
