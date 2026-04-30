## NRE Metrics Uploader · Kratos Telemetry

Upload `metrics.yaml` data to the SQA Kratos Telemetry database.

### 1) Prerequisites

- Python 3.11+
- Bazel

Environment variables (required):

```bash
export NRE_SQA_SSA_URL=<url>
export NRE_SQA_SSA_CLIENT_ID=<client_id>
export NRE_SQA_SSA_CLIENT_SECRET=<client_secret>
export NRE_SQA_KRATOS_TELEMETRY_ENDPOINT=<endpoint>
export NRE_SQA_KRATOS_SCHEMAID=<schema_id>
```

_Note: These secrets are maintained and administered by SQA. Please contact [@tghare](https://nvidia.enterprise.slack.com/archives/D03RBBFRYG7) or drop a message in the **[#swdl-nre-sqa](https://nvidia.enterprise.slack.com/archives/C08A08B9EFL)** slack channel to get access._

### 2) Set up s3cmd (once)

```bash
bazel run @nre_pip_deps//:pip -- install s3cmd     # or: sudo apt-get install s3cmd
python3 -m s3cmd --configure                       # add AWS creds for the bucket with metrics files
python3 -m s3cmd ls s3://<S3_bucket_name>/         # verify access
```

_Note: The AWS creds (including S3 API key/secret) can be retrieved from NVIDIA CSS portal, under the auth info for **team-ncore** namespace._

### 3) Quick start

- Upload a single metrics file (two ways):

  - Local file path:

  ```bash
  bazel run //internal/scripts/kratos:upload_metrics_to_kratos -- /path/to/metrics.yaml
  ```

  Fetches the metrics.yaml, validates the format/structure, and uploads them in batches.

  - S3 URL (auto-downloads via s3cmd then uploads):

  ```bash
  bazel run //internal/scripts/kratos:upload_metrics_to_kratos -- s3://<bucket>/path/to/metrics.yaml
  ```

  Scans the file path from S3 bucket, downloads them to a temporary storage, validates the format/structure, and uploads them in batches.

- Monitor a bucket and upload new files automatically (bucket required):

  ```bash
  bazel run //internal/scripts/kratos:metrics_monitor -- --bucket <S3_bucket_name>
  ```

  Lists the S3 objects via cmd, scans and filters the file path, skips historically processed files, downloads new ones to a temporary storage, validates the format/structure, and uploads them in batches.

#### To check all the available options, run the below helper commands:

````bash
bazel run //internal/scripts/kratos:upload_metrics_to_kratos -- --help  # Helper CLI for one-off uploads.```

bazel run //internal/scripts/kratos:metrics_monitor -- --help           # Helper CLI for monitoring and validation.```

bazel run //internal/scripts/kratos:setup_monitor -- --help             # Helper CLI for ops workflows (check, scan, scan-only, dry-run, stats, reset-state, one).```
````

### 4) Troubleshooting

- s3cmd not configured: run `s3cmd --configure`
- 403 Forbidden: check AWS credentials and bucket permissions
- Upload failed: re-check the NRE*SQA*\* environment variables
- State file permission errors: change `--state-file` path or fix permissions

Diagnostics:

```bash
bazel run //internal/scripts/kratos:metrics_monitor -- --verbose
bazel run //internal/scripts/kratos:setup_monitor -- check --verbose
```
