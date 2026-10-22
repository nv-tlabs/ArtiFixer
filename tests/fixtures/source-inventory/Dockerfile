# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
ARG REVISION
COPY install_metadata.py /tmp/install_metadata.py
RUN python /tmp/install_metadata.py "${REVISION}"
# Isolate source transport, not inventory/selection: the real collector still
# reads installed direct_url.json, compares inventories and creates its archive.
COPY sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py
# The two fixtures have identical dpkg inventories, so there are no apt sources
# to retrieve. Suppress the collector's otherwise unconditional apt refresh.
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/apt-get && chmod +x /usr/local/bin/apt-get
