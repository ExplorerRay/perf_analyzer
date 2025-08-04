FROM ubuntu:24.04

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get -y install \
  python3-pip \
 && apt-get clean autoclean \
 && apt-get autoremove --yes \
 && rm -rf /var/lib/apt/lists/*

RUN pip install perf_analyzer --break-system-packages

COPY ./genai-perf /genai-perf/

RUN pip install /genai-perf --break-system-packages

ENTRYPOINT ["genai-perf", "config", "-f", "/aiperf/config.yml", "--override-config"]
# CMD ["/bin/bash"]
