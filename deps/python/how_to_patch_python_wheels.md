# How to Patch Python Wheels

## Introduction

This guide provides step-by-step instructions for patching Python wheels. This is useful when you need to modify a package without waiting for upstream changes.

## Prerequisites

### 1. Create a conda environment

```bash
conda create -n patch_wheel_env python=3.11 -y
```

### 2. Activate the environment

```bash
conda activate patch_wheel_env
```

## Steps

### 1. Download the original wheel

```bash
pip download package==version --python-version 3.11 --only-binary=:all: --no-deps
```

### 2. Extract the wheel contents

```bash
unzip package-version-py3-none-any.whl -d patched
```

### 3. Apply your patch

- Either copy files from your patched repository:
  ```bash
  rsync -avr ../patched-repo/path/to/module/* patched/path/to/module/
  ```
- Or manually edit the necessary files

### 4. Update version information

- Get the git hash of your patch:
  ```bash
  GIT_HASH=$(git -C patched-repo rev-parse --short HEAD)
  NEW_VERSION="version+git$GIT_HASH"
  ```
- Update the METADATA file:
  ```bash
  sed -i "s/original-version/$NEW_VERSION/" patched/package-version.dist-info/METADATA
  ```

### 5. Rename the dist-info directory

```bash
mv patched/package-version.dist-info patched/package-$NEW_VERSION.dist-info
```

### 6. Repack the wheel

```bash
python -m wheel pack patched
```

### 7. Verify the patched wheel (optional)

```bash
unzip -l package-$NEW_VERSION-py3-none-any.whl
```

### 8. Publish the wheel

```bash
pip install twine
twine upload --repository your-repo-url package-$NEW_VERSION-py3-none-any.whl
```

## Example

```bash
# Create and activate conda environment
conda create -n package_patch_env python=3.11 -y
conda activate package_patch_env

# Install necessary tools
pip install wheel twine

# Download and patch the wheel
pip download setuptools==78.1.1 --python-version 3.11 --only-binary=:all: --no-deps
unzip setuptools-78.1.1-py3-none-any.whl -d patched
rsync -avr ../setuptools/setuptools/_vendor/jaraco/* patched/setuptools/_vendor/jaraco/

# Update version and repack
GIT_HASH=$(git -C ../setuptools rev-parse --short HEAD)
NEW_VERSION="78.1.1+git$GIT_HASH"
sed -i "s/78.1.1/$NEW_VERSION/" patched/setuptools-78.1.1.dist-info/METADATA
mv patched/setuptools-78.1.1.dist-info patched/setuptools-$NEW_VERSION.dist-info
python -m wheel pack patched

# Upload the patched wheel
twine upload --repository your-repo-url setuptools-$NEW_VERSION-py3-none-any.whl

# Deactivate the environment when done
conda deactivate
```

## Notes

- When using `python -m wheel pack`, the RECORD file is automatically updated
- The WHEEL file typically doesn't contain version information that needs updating
- For automation, consider creating a script or Dockerfile to handle the process
- Version naming convention: `<original_version>+git<git_hash>` (e.g., 78.1.1+gitb18a11365)
