# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Script to download models from model registries."""

import argparse
import logging
import os
import sys

from pathlib import Path

from nre.utils.model_registry import ModelRegistryError, create_model_registry, log_and_raise


logger = logging.getLogger(__name__)


def main():
    """Main entry point for the model download script."""
    parser = argparse.ArgumentParser(
        description="Download a model from supported model registries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download from NGC using API key from command line:
    bazel run //internal/scripts/ngc:download_model -- \\
        --url https://api.ngc.nvidia.com/v2/org/nvidian/team/ct-toronto-ai/models/test-difix-model-nre-av/versions/e4253f9f/files/difix_pretrained_model-e4253f9f.tar.gz \\
        --output-dir /tmp/ \\
        --api-key your-api-key

    # Download from GitLab using API key from environment:
    export MODEL_API_KEY=your-api-key
    bazel run //internal/scripts/ngc:download_model -- \\
        --url https://gitlab-master.nvidia.com/api/v4/projects/85874/packages/generic/difix_pretrained_models/3.0/difix_pretrained_model-e4253f9f.tar.gz \\
        --output-dir /tmp/

    # Download using API key from .netrc:
    # Add this to your ~/.netrc file:
    # machine api.ngc.nvidia.com
    #     password your-api-key
    # OR
    # machine gitlab-master.nvidia.com 
    #     password your-api-key
    bazel run //internal/scripts/ngc:download_model -- \\
        --url https://api.ngc.nvidia.com/v2/org/nvidian/team/ct-toronto-ai/models/test-difix-model-nre-av/versions/e4253f9f/files/difix_pretrained_model-e4253f9f.tar.gz \\
        --output-dir /tmp/
        """,
    )
    parser.add_argument(
        "--url",
        required=True,
        help="URL to download from (supported domains: api.ngc.nvidia.com, gitlab-master.nvidia.com)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Directory to save the downloaded model (default: models)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the model registry, default is None",
    )

    args = parser.parse_args()

    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Created output directory: %s", output_dir)

        # Initialize the appropriate registry using the factory
        logger.info("Initializing model registry for URL: %s", args.url)
        registry = create_model_registry(
            model_url=args.url,
            model_cache_dir=output_dir,
            api_key=args.api_key,  # Registry will try environment variable NGC_API_KEY or .netrc if this is None
        )

        # Download the model
        _ = registry.get_model()

    except ModelRegistryError as e:
        log_and_raise(ModelRegistryError, f"Failed to download model: {str(e)}")
    except Exception as e:
        log_and_raise(Exception, f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
