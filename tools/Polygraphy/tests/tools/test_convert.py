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
import tempfile
from textwrap import dedent

import onnx
import pytest
import tensorrt as trt
from polygraphy import mod
from polygraphy.backend.common.loader import BytesFromPath
from polygraphy.backend.trt.loader import EngineFromBytes
from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.tools.common import run_polygraphy_convert


class TestConvertToOnnx(object):
    def test_tf2onnx(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as outmodel:
            run_polygraphy_convert([TF_MODELS["identity"].path, "--model-type=frozen", "-o", outmodel.name])
            assert onnx.load(outmodel.name)


    def test_fp_to_fp16(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_convert([ONNX_MODELS["identity_identity"].path, "--convert-to=onnx", "--fp-to-fp16", "-o", outmodel.name])
            assert onnx.load(outmodel.name).graph.value_info[0].type.tensor_type.elem_type == 10


class TestConvertToTrt(object):
    def check_engine(self, path):
        loader = EngineFromBytes(BytesFromPath(path))
        with loader() as engine:
            assert isinstance(engine, trt.ICudaEngine)


    def test_onnx_to_trt(self):
        with tempfile.NamedTemporaryFile(suffix=".engine") as outmodel:
            run_polygraphy_convert([ONNX_MODELS["identity"].path, "--model-type=onnx", "-o", outmodel.name])
            self.check_engine(outmodel.name)


    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Bug in older versions of TRT breaks this test")
    def test_tf_to_onnx_to_trt(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_convert([TF_MODELS["identity"].path, "--model-type=frozen", "--convert-to=trt", "-o", outmodel.name])
            self.check_engine(outmodel.name)


    def test_trt_network_config_script_to_engine(self):
        script = dedent("""
        from polygraphy.backend.trt import CreateNetwork, CreateConfig
        from polygraphy import func
        import tensorrt as trt

        @func.extend(CreateNetwork())
        def my_load_network(builder, network):
            inp = network.add_input("input", dtype=trt.float32, shape=(1, 1))
            out = network.add_identity(inp).get_output(0)
            network.mark_output(out)

        @func.extend(CreateConfig())
        def load_config(config):
            config.set_flag(trt.BuilderFlag.FP16)
        """)

        with tempfile.NamedTemporaryFile("w+", suffix=".py") as f, tempfile.NamedTemporaryFile() as outmodel:
            f.write(script)
            f.flush()

            run_polygraphy_convert([f.name, "--model-type=trt-network-script", "--trt-network-func-name=my_load_network", "--trt-config-script", f.name,
                                    "--convert-to=trt", "-o", outmodel.name])
            self.check_engine(outmodel.name)


    def test_modify_onnx_outputs(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as outmodel:
            run_polygraphy_convert([ONNX_MODELS["identity_identity"].path, "-o", outmodel.name, "--onnx-outputs", "mark", "all"])

            model = onnx.load(outmodel.name)
            assert len(model.graph.output) == 2
