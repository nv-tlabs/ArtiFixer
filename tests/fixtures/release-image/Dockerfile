# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
RUN printf '\177ELFfixture-old-codec-bytes' > /usr/local/lib/libavcodec.so.fixture
RUN rm /usr/local/lib/libavcodec.so.fixture
# Deliberate test doubles: this fixture tests the release workflow, not OpenCV.
RUN printf '__version__ = "fixture"\ndef getBuildInformation(): return "FFMPEG: NO"\n' > /usr/local/lib/python3.12/site-packages/cv2.py && printf 'def get_ffmpeg_exe(): raise RuntimeError("No FFmpeg")\n' > /usr/local/lib/python3.12/site-packages/imageio_ffmpeg.py
ENV EMPTY="" LITERAL="\$HOME \"quoted\" \\path"
LABEL fixture="metadata-only release workflow test"
WORKDIR /directory with spaces
USER 1001:1002
EXPOSE 8888 5000/udp
STOPSIGNAL SIGTERM
SHELL ["/bin/sh", "-e", "-c"]
ENTRYPOINT ["python"]
CMD ["-c", "print('configuration intact')"]
