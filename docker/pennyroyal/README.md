# Pennyroyal container

This Compose service runs the same Pennyroyal v2.5.0 source and launch recipes
as the native installation. The default is Flash-Next with FR-Spec. Native
installation remains supported and is documented in [`BUILD.md`](../../BUILD.md)
and [`RUN.md`](../../RUN.md).

The image is `ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.0`, built and uploaded
by GitHub Actions. The download is approximately **8.43 GiB**, excluding models.
Python, the CUDA toolchain, NIXL POSIX, and prebuilt FlashInfer kernels are
included; the host supplies the NVIDIA driver. Existing native installations
do not need to change.

Both profiles passed ordinary API schema/tool checks, 64K prefill, 1,024-token
C1/C4 decode, a JPEG spatial check, a static-video frame-path check, and NIXL
reuse after container restart. Each restored 63,872 of 63,906 prompt tokens
from storage and returned exact `READY`. GPU serving was checked with rootless
Podman on one RTX PRO 6000; the supplied Docker Compose configuration was
checked separately. These are container packaging checks, not new model-quality
scores or a performance-improvement claim. The Next check used the RadixArk
reference target; 27B used the measured FP8 checkpoint identified in
[`BUILD.md`](../../BUILD.md#reference-and-measured-checkpoints).

## Prerequisites

- Linux x86-64, ordinary rootful Docker Engine with Compose v2, an NVIDIA driver
  compatible with the image's CUDA 13.3 toolkit, and the NVIDIA
  Container Toolkit configured for Docker.
- An RTX PRO 6000 Blackwell and the model files described in
  [`BUILD.md`](../../BUILD.md#reference-and-measured-checkpoints).
- Writable host directories for compiler/runtime caches and NIXL persistence.
  The runtime UID and GID must own them.
- A local filesystem suitable for NIXL POSIX O_DIRECT/io_uring storage.

The container does not reduce host RAM requirements. The Next recipe uses a
32 GiB HiCache tier plus roughly 48 GiB for RAM-backed PLE; the 27B recipe
reserves a 96 GiB HiCache tier. Leave additional room for loading, the runtime
and the operating system. See [host memory and first start](../../RUN.md#host-memory-and-first-start);
NVMe PLE is an optional way to reduce Next's host-memory use.

The UID/GID examples below assume Docker without `userns-remap`. Rootless
Docker and remapped daemons use different host/container UID mappings; adapt
bind-directory ownership to that mapping instead of copying the example
ownership commands unchanged. Do not switch off host-wide user namespaces
just to use this recipe.

The service is not privileged. It does use `seccomp=unconfined`, because the
NIXL POSIX path needs io_uring and Docker's default seccomp profile commonly
blocks it. If your daemon uses a custom profile that explicitly permits the
required io_uring syscalls, replace this setting with that profile.

## Configure and start

Get the launch files from the current public branch (the original
v2.5.0 source tag predates container packaging):

```bash
git clone --depth 1 --branch pennyroyal-main-sm120-final \
  https://github.com/jpezzulli/sglang-rtxpro6000.git pennyroyal
cd pennyroyal/docker/pennyroyal
cp .env.example .env
```

This checkout supplies configuration and documentation; Docker pulls the
prebuilt image. You do not build SGLang locally.

Edit `.env` with the three host roots and the model paths visible below
`/models`. Mount a common parent as `HOST_MODELS_ROOT` when checkpoint files
are symlinks to siblings; links that escape the bind mount will be broken.
Create the writable directories with the configured numeric identity, for
example:

```bash
sudo install -d -o 1000 -g 1000 \
  /var/cache/pennyroyal /srv/pennyroyal-nixl
docker compose pull
docker compose up
```

The API is published at `http://localhost:8001/v1` by default. Change
`PENNYROYAL_PORT` for a different host port. Startup can legitimately take
many minutes while weights, extensions, graphs, and caches initialize; the
image health check allows a 20-minute start period. A bad configuration exits
instead of entering an automatic restart loop.

Run in the background and inspect it with:

```bash
docker compose up -d
docker compose logs -f pennyroyal
docker compose ps
docker compose down
```

`down` allows up to two minutes for shutdown, then removes the container and
network, not the three bind-mounted host directories.

## Profiles and checks

Set `PENNYROYAL_PROFILE` in `.env` to one of:

| Value | Recipe |
|---|---|
| `next` | Flash-Next with FR-Spec (default) |
| `next-plain` | Flash-Next without FR-Spec |
| `27b` | Qwen3.8-27B target with the DFlash2 draft |

For 27B, change **all three** settings in `.env`, using your actual downloaded
directory names below `/models`:

```dotenv
PENNYROYAL_PROFILE=27b
TARGET_MODEL=/models/Qwen3.8-27B-FP8
DRAFT_MODEL=/models/Qwen3.8-27B-DFlash2
```

Changing the profile alone leaves the example's Flash-Next target selected;
it does not automatically choose or download a 27B checkpoint. The two Next
profiles use only `TARGET_MODEL`.

The entrypoint also exposes two non-serving checks. The import check is
CPU-only and deliberately skips device work; neither command qualifies GPU
serving:

```bash
docker run --rm ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.0 --help
docker run --rm ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.0 --check
```

Arbitrary commands require the explicit `exec` boundary:

```bash
docker compose run --rm pennyroyal exec .venv/bin/python --version
```

## Optional settings

Online FP8 is off by default. Read [`FP8.md`](../../FP8.md), then set
`SGLANG_SM120_ONLINE_MXFP8=true` to opt in. RAM-backed PLE is the default.

For NVMe-backed PLE, read [`NVME-PLE.md`](../../NVME-PLE.md). The image already
contains the isolated reader, but a prepared overlay is still required. The
normal `/models` mount is deliberately read-only. For the one-time preparation
only, override that mount as writable and create a new output directory under
it (the helper refuses to overwrite an existing output):

```bash
docker compose run --rm --no-deps \
  -v /srv/models:/models \
  pennyroyal exec .venv/bin/python scripts/pennyroyal/prepare_ple_nvme.py \
  --source /models/RadixArk-Qwen3.8-Flash-Next-NVFP4 \
  --output /models/flash-next-ple
```

Use the actual `HOST_MODELS_ROOT` in place of `/srv/models`. Keep the source
checkpoint immutable. After preparation, restore the normal read-only mount,
set `PENNY_PLE_BACKEND=nvme` and `PENNY_PLE_NVME_MODEL=/models/flash-next-ple`,
then start either Next profile. NVMe PLE is not supported by the 27B profile.

To use a second GPU only for media preprocessing, create `compose.override.yaml`:

```yaml
services:
  pennyroyal:
    environment:
      SGLANG_MM_PREPROCESS_DEVICE: cuda:1
    deploy:
      resources:
        reservations:
          devices: !override
            - driver: nvidia
              device_ids: ["0", "1"]
              capabilities: [gpu]
```

Inside the container, `cuda:0` remains the model GPU and `cuda:1` is the
secondary preprocessing GPU. `!override` requires Docker Compose 2.24.4 or
newer and replaces, rather than appends to, the default device reservation.
This does not split the model. Use GPU UUIDs in `device_ids` when stable device
selection matters.

### SELinux hosts

If your container engine enables SELinux confinement, ordinary UID/GID
ownership may not be enough to read the bind mounts. Rather than recursively
relabeling a large model directory shared with native services, add this
per-container override to `compose.override.yaml`:

```yaml
services:
  pennyroyal:
    security_opt:
      - label=disable
```

This disables SELinux separation for this container only; it does not disable
host SELinux, make the container privileged, or change the read-only model
mount. Omit it when your engine does not enable SELinux confinement.
