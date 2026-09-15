# Pennyroyal container

This Compose service runs the same Pennyroyal v2.5.0 source and launch recipes
as the native installation. The default is Flash-Next with FR-Spec. Native
installation remains supported and is documented in [`BUILD.md`](../../BUILD.md)
and [`RUN.md`](../../RUN.md).

The default image is
`ghcr.io/jpezzulli/sglang-rtxpro6000:v2.5.0`. It is an initial container
delivery pending runtime qualification; its tag alone is not evidence that the
container path has passed the native release's model and performance checks.
The image can be built in CPU-only GitHub Actions, but serving requires the
supported NVIDIA GPU environment.

## Prerequisites

- Linux, Docker Engine with Compose v2, an NVIDIA driver, and the NVIDIA
  Container Toolkit configured for Docker.
- An RTX PRO 6000 Blackwell and the model files described in
  [`BUILD.md`](../../BUILD.md#reference-and-measured-checkpoints).
- Writable host directories for compiler/runtime caches and NIXL persistence.
  The runtime UID and GID must own them.
- A local filesystem suitable for NIXL POSIX O_DIRECT/io_uring storage.

The service is not privileged. It does use `seccomp=unconfined`, because the
NIXL POSIX path needs io_uring and Docker's default seccomp profile commonly
blocks it. If your daemon uses a custom profile that explicitly permits the
required io_uring syscalls, replace this setting with that profile.

## Configure and start

Work from this directory:

```bash
cd docker/pennyroyal
cp .env.example .env
```

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

`down` removes the container and network, not the three bind-mounted host
directories.

## Profiles and checks

Set `PENNYROYAL_PROFILE` in `.env` to one of:

| Value | Recipe |
|---|---|
| `next` | Flash-Next with FR-Spec (default) |
| `next-plain` | Flash-Next without FR-Spec |
| `27b` | Qwen3.8-27B target with the DFlash2 draft |

The 27B profile requires both `TARGET_MODEL` and `DRAFT_MODEL`. The two Next
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
