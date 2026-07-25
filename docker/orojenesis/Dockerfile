# syntax=docker/dockerfile:1.7

ARG OROJENESIS_BUILDER_IMAGE=ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90
FROM ${OROJENESIS_BUILDER_IMAGE} AS builder

ARG OROJENESIS_COMMIT=97d52178bf9a9c209bf79be96b87c164bcd35625
ARG OROJENESIS_TREE_OID=05b05ec5a2a2979b1fe92046b937556d9ad99847
ARG OROJENESIS_SOURCE_ARCHIVE_SHA256=3a254ab201d92b7eba993d3c7dcf0bb148a31dc9e57ece020fbaa38ad67c7873
ARG OROJENESIS_SOURCE_DATE_EPOCH=1753058729
ARG UBUNTU_SNAPSHOT=20260718T000000Z
ARG OROJENESIS_BUILDER_IMAGE
ENV DEBIAN_FRONTEND=noninteractive

RUN test "$(dpkg --print-architecture)" = "amd64"
RUN apt-get --snapshot "${UBUNTU_SNAPSHOT}" update \
    && apt-get --snapshot "${UBUNTU_SNAPSHOT}" install -y --no-install-recommends \
      build-essential ca-certificates cmake git ninja-build python3 scons \
      libconfig++-dev libboost-dev libboost-iostreams-dev \
      libboost-serialization-dev libyaml-cpp-dev libncurses-dev \
      libtinfo-dev libgpm-dev libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --chmod=0755 docker/orojenesis/g++-cstdint /tmp/orojenesis-compiler/g++
COPY scripts/internal/orojenesis/write_provenance.py /tmp/write_provenance.py

RUN git clone https://github.com/NVlabs/timeloop.git /opt/orojenesis \
    && git -C /opt/orojenesis checkout --detach "${OROJENESIS_COMMIT}" \
    && test "$(git -C /opt/orojenesis rev-parse HEAD)" = "${OROJENESIS_COMMIT}" \
    && test "$(git -C /opt/orojenesis rev-parse HEAD^{tree})" = "${OROJENESIS_TREE_OID}" \
    && git -C /opt/orojenesis archive --format=tar HEAD > /tmp/orojenesis.tar \
    && echo "${OROJENESIS_SOURCE_ARCHIVE_SHA256}  /tmp/orojenesis.tar" | sha256sum -c - \
    && ln -s ../pat-public/src/pat /opt/orojenesis/src/pat \
    && SOURCE_DATE_EPOCH="${OROJENESIS_SOURCE_DATE_EPOCH}" \
       PATH="/tmp/orojenesis-compiler:${PATH}" \
       scons -C /opt/orojenesis --static -j4 \
    && test -x /opt/orojenesis/bin/timeloop-mapper \
    && python3 /tmp/write_provenance.py \
         --home /opt/orojenesis \
         --expected-commit "${OROJENESIS_COMMIT}" \
         --expected-tree "${OROJENESIS_TREE_OID}" \
         --expected-archive-sha256 "${OROJENESIS_SOURCE_ARCHIVE_SHA256}" \
         --builder-image "${OROJENESIS_BUILDER_IMAGE}" \
         --ubuntu-snapshot "${UBUNTU_SNAPSHOT}" \
         --source-date-epoch "${OROJENESIS_SOURCE_DATE_EPOCH}" \
         --compiler-wrapper /tmp/orojenesis-compiler/g++ \
         --output /opt/orojenesis/orojenesis-provenance.json

FROM ${OROJENESIS_BUILDER_IMAGE} AS artifact

ARG UBUNTU_SNAPSHOT=20260718T000000Z
RUN apt-get --snapshot "${UBUNTU_SNAPSHOT}" update \
    && apt-get --snapshot "${UBUNTU_SNAPSHOT}" install -y --no-install-recommends \
      libboost-iostreams1.83.0 libboost-serialization1.83.0 \
      libconfig++9v5 libgmp10 libyaml-cpp0.8 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/orojenesis /opt/orojenesis
ENV SOLAR_OROJENESIS_HOME=/opt/orojenesis
ENTRYPOINT ["/opt/orojenesis/bin/timeloop-mapper"]
