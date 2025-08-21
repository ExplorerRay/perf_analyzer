# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import genai_perf.logging as logging
import requests
import yaml


def load_config(config_path):
    """Load the configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def generate_combinations(config) -> dict:
    """Generate all combinations of model and token configurations."""
    models = config.get("models", [])
    token_confs = config.get("token_confs", {})
    inputs = token_confs.get("input", [])
    input_mean_stddev = [(i.get("mean"), i.get("stddev")) for i in inputs]
    outputs = token_confs.get("output", [])
    output_mean_stddev = [(o.get("mean"), o.get("stddev")) for o in outputs]
    reqs = config.get("requests", {})
    run_counts = reqs.get("run_count", [0])
    concurrency = config.get("concurrency", [1])

    model_combinations = {}
    for model in models:
        model_combinations[model] = product(
            input_mean_stddev,
            output_mean_stddev,
            run_counts,
            concurrency,
        )

    return model_combinations


def warmup_request(url, model, warmup_id):
    header = {"Content-Type": "application/json"}
    warmup_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Warmup {warmup_id + 1}"},
        ],
    }
    try:
        response = requests.post(url, headers=header, json=warmup_data)
        response.raise_for_status()
        return
    except requests.RequestException as e:
        print(f"Warmup request failed for warmup {warmup_id + 1}: {e}")
        exit(1)


def custom_warmup(url, model, warmup_count, concurrency):
    """
    Custom warmup for loading model to RAM (avoid cold-start)
    with concurrency, without streaming, no response
    if concurrency > warmup count, the max concurrency will be limited to warmup count
    """
    req_url = f"http://{url}/v1/chat/completions"
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for w in range(warmup_count):
            executor.submit(warmup_request, req_url, model, w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and render configuration from a YAML file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    itpe_perf_conf = config.get("itpe_perf", {})
    model_combinations = generate_combinations(itpe_perf_conf)
    logging.init_logging()
    logger = logging.getLogger(__name__)

    # iterate models
    for model, combinations in model_combinations.items():
        for c in combinations:
            endpoint_url = itpe_perf_conf.get("url", "")

            # custom warmup request(s) for loading model to RAM
            if c[3] > 0:
                logger.info(
                    f"Start running warmup for {model} with {c[3]} requests and {c[3]} concurrency"
                )
                custom_warmup(endpoint_url, model, c[3], c[3])
                logger.info("Warmup completed")

            # Set options for GenAI perf
            cmd = [
                "genai-perf",
                "profile",
                "--url",
                str(endpoint_url),
                "--model",
                model,
                "--synthetic-input-tokens-mean",
                str(c[0][0]),
                "--synthetic-input-tokens-stddev",
                str(c[0][1]),
                "--output-tokens-mean",
                str(c[1][0]),
                "--output-tokens-stddev",
                str(c[1][1]),
                "--request-count",
                str(c[2]),
                "--concurrency",
                str(c[3]),
                "--profile-export-file",
                f"{c[2]}_{c[3]}_profile.json",
                "--endpoint-type",
                "chat",
                "--tokenizer",
                "hf-internal-testing/llama-tokenizer",
                "--artifact-dir",
                "/artifacts",
            ]
            if itpe_perf_conf.get("enabled", {}).get("stream", False):
                cmd.append("--streaming")
            if itpe_perf_conf.get("enabled", {}).get("checkpoint", False):
                cmd.append("--enable-checkpointing")
                cmd.append("--checkpoint-dir")
                cmd.append("/artifacts")
            # Run GenAI perf
            subprocess.run(cmd)
