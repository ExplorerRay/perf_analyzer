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
import os
import subprocess

import jinja2
import yaml

TEMP_CONF_PATH = "/tmp/rendered_config.yml"


def load_config(config_path):
    """Load the configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


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
    parser.add_argument(
        "--template",
        type=str,
        default="config.yml.j2",
        help="Path to the Jinja2 template file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.template, "r") as template_file:
        template_content = template_file.read()
    template = jinja2.Template(template_content)

    # iterate models
    index = 0
    for model in config.get("models", []):
        # iterate token_confs (input/output #tokens)
        inputs = config.get("token_confs", {}).get("input", [])
        outputs = config.get("token_confs", {}).get("output", [])
        for i in inputs:
            for o in outputs:
                # render the template with the current model and token configuration
                rendered_config = template.render(
                    model=model,
                    synthetic_mean=str(i.get("mean")),
                    synthetic_stddev=str(i.get("stddev")),
                    output_mean=str(o.get("mean")),
                    output_stddev=str(o.get("stddev")),
                )

                # Save the rendered configuration to a temp file and execute it
                with open(TEMP_CONF_PATH, "w") as f:
                    f.write(rendered_config)
                subprocess.run(
                    ["genai-perf", "config", "-f", TEMP_CONF_PATH],
                )

    # Clean up the temporary file
    if os.path.exists(TEMP_CONF_PATH):
        os.remove(TEMP_CONF_PATH)
