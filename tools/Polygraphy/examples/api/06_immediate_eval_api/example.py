#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This script uses Polygraphy's immediately evaluated functional APIs
to load an ONNX model, convert it into a TensorRT network, add an identity
layer to the end of it, build an engine with FP16 mode enabled, and finally
run inference.
"""
import numpy as np

from polygraphy.backend.trt import (TrtRunner, create_config,
                                    engine_from_network,
                                    network_from_onnx_path)


def main():
    # In Polygraphy, loaders and runners take ownership of objects if they are provided
    # via the return values of callables. For example, we don't need to worry about object
    # lifetimes when we use lazy loaders.
    #
    # Since we are immediately evaluating, we take ownership of objects, and are responsible for freeing them.
    builder, network, parser = network_from_onnx_path("identity.onnx")

    # Extend the network with an identity layer.
    prev_output = network.get_output(0)
    network.unmark_output(prev_output)
    output = network.add_identity(prev_output).get_output(0)
    output.name = "output"
    network.mark_output(output)

    # Create a TensorRT IBuilderConfig so that we can build the engine with FP16 enabled.
    config = create_config(builder, network, fp16=True)

    # We can free everything we constructed above once we're done building the engine.
    # NOTE: In TensorRT 8.0, we do *not* need to use a context manager here.
    with builder, network, parser, config:
        engine = engine_from_network((builder, network), config)

    # NOTE: In TensorRT 8.0, we do *not* need to use a context manager to free `engine`.
    with engine, TrtRunner(engine) as runner:
        inp_data = np.ones((1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.all(outputs["output"] == inp_data) # It's an identity model!


if __name__ == "__main__":
    main()
