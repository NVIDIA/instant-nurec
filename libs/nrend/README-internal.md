# NREnd for Omniverse

How to publish nrend (and tcnn) packages to packman:

### Short Version:

1. Build docker container
2. Run container, copy package(s) to local drive
3. Upload to packman

### Detailed Version:

1. Find your GITLAB_TOKEN and build the docker container in `libs/nrend`. This will build tcnn and slang in centos.
   For both tcnn it uses the branch `publish/nrend` which contains some packman and repoman utilities.
   For slang it uses tag v2025.4 which contains the correct version expected by nrend.
   If the docker build uses the cache despite repo changes, modify the `ARG CACHEBUST_XXXX` value to force a new build.

```
nre/libs/nrend$ GITLAB_TOKEN=SpecifyYourToken ./docker_update_nrend_deps.sh
```

2. Run the docker container and build nrend inside it.

```
nre/libs/nrend$ ./docker_run_build_nrend.sh
```

3. The packman packages are in `/tmp/`. Make sure the package(s) look correct and have the required files. See below for how to test locally. The filename should look something like this: `nrend@1.1.0+cu118-linux-x86_64.7z`.
   To upload manually, go to `https://omnipackages.nvidia.com/`, login, click publish (cloudfront) and upload the file.

### Local Testing

To test the package in an omniverse project, find the line in `target-deps.packman.xml` including the package.

```
  <package name="nrend" version="1.1.0+cu118-${platform}" platforms="linux-x86_64" />
```

Change to this with the correct relative path:

```
  <source path="../../relative/path/to/unzipped/nrend/package" />
```
