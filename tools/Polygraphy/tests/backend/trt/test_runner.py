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
import threading

import numpy as np
import pytest
import tensorrt as trt
from polygraphy import cuda, mod
from polygraphy.backend.trt import (CreateConfig, EngineFromNetwork,
                                    NetworkFromOnnxBytes, Profile, TrtRunner,
                                    engine_from_network)
from polygraphy.exception import PolygraphyException
from polygraphy.logger import G_LOGGER
from tests.models.meta import ONNX_MODELS


class TestLoggerCallbacks(object):
    @pytest.mark.parametrize("sev", G_LOGGER.SEVERITY_LETTER_MAPPING.keys())
    def test_set_severity(self, sev):
        G_LOGGER.severity = sev


class TestTrtRunner(object):
    def test_can_name_runner(self):
        NAME = "runner"
        runner = TrtRunner(None, name=NAME)
        assert runner.name == NAME


    def test_basic(self):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            assert runner.is_active
            assert runner.owns_engine
            assert runner.owns_context
            model.check_runner(runner)
        assert not runner.is_active
        assert runner._cached_input_metadata is None


    def test_context(self):
        model = ONNX_MODELS["identity"]
        engine = engine_from_network(NetworkFromOnnxBytes(model.loader))
        with engine, TrtRunner(engine.create_execution_context) as runner:
            model.check_runner(runner)
            assert not runner.owns_engine
            assert runner.owns_context


    def test_device_buffer_order_matches_bindings(self):
        model = ONNX_MODELS["reducable"]
        engine = engine_from_network(NetworkFromOnnxBytes(model.loader))
        with engine, TrtRunner(engine) as runner:
            dev_buf_order = list(runner.device_buffers.keys())
            for binding, dev_buf_name in zip(engine, dev_buf_order):
                assert binding == dev_buf_name


    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_shape_output(self):
        model = ONNX_MODELS["reshape"]
        engine = engine_from_network(NetworkFromOnnxBytes(model.loader))
        with engine, TrtRunner(engine.create_execution_context) as runner:
            model.check_runner(runner)


    def test_multithreaded_runners_from_engine(self):
        model = ONNX_MODELS["identity"]
        engine = engine_from_network(NetworkFromOnnxBytes(model.loader))

        with engine, TrtRunner(engine) as runner0, TrtRunner(engine) as runner1:
            t1 = threading.Thread(target=model.check_runner, args=(runner0, ))
            t2 = threading.Thread(target=model.check_runner, args=(runner1, ))
            t1.start()
            t2.start()
            t1.join()
            t2.join()


    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_multiple_profiles(self):
        model = ONNX_MODELS["dynamic_identity"]
        profile0_shapes = [(1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4)]
        profile1_shapes = [(1, 2, 4, 4), (1, 2, 8, 8), (1, 2, 16, 16)]
        network_loader = NetworkFromOnnxBytes(model.loader)
        profiles = [
            Profile().add("X", *profile0_shapes),
            Profile().add("X", *profile1_shapes),
        ]
        config_loader = CreateConfig(profiles=profiles)
        with TrtRunner(EngineFromNetwork(network_loader, config_loader)) as runner:
            for index, shapes in enumerate([profile0_shapes, profile1_shapes]):
                runner.set_profile(index)
                for shape in shapes:
                    model.check_runner(runner, {"X": shape})


    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    def test_empty_tensor_with_dynamic_input_shape_tensor(self):
        model = ONNX_MODELS["empty_tensor_expand"]
        shapes = [(1, 2, 0, 3, 0), (2, 2, 0, 3, 0), (4, 2, 0, 3, 0)]
        network_loader = NetworkFromOnnxBytes(model.loader)
        profiles = [Profile().add("new_shape", *shapes)]
        config_loader = CreateConfig(profiles=profiles)

        with TrtRunner(EngineFromNetwork(network_loader, config_loader)) as runner:
            for shape in shapes:
                model.check_runner(runner, {"new_shape": shape})


    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Test not compatible with TRT 6")
    @pytest.mark.parametrize("names, err", [
        (["fake-input", "x"], "Extra keys in"),
        (["fake-input"], "Some keys are missing"),
        ([], "Some keys are missing"),
    ])
    def test_error_on_wrong_name_feed_dict(self, names, err):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            with pytest.raises(PolygraphyException, match=err):
                runner.infer({name: np.ones(shape=(1, 1, 2, 2), dtype=np.float32) for name in names})


    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Test not compatible with TRT 6")
    def test_error_on_wrong_dtype_feed_dict(self):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            with pytest.raises(PolygraphyException, match="unexpected dtype."):
                runner.infer({"x": np.ones(shape=(1, 1, 2, 2), dtype=np.int32)})


    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Test not compatible with TRT 6")
    def test_error_on_wrong_shape_feed_dict(self):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            with pytest.raises(PolygraphyException, match="incompatible shape."):
                runner.infer({"x": np.ones(shape=(1, 1, 3, 2), dtype=np.float32)})


    @pytest.mark.parametrize("use_view", [True, False]) # We should be able to use DeviceArray in place of DeviceView
    def test_device_views(self, use_view):
        model = ONNX_MODELS["reducable"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner, cuda.DeviceArray((1, ), dtype=np.float32) as x:
            x.copy_from(np.ones((1, ), dtype=np.float32))
            outputs = runner.infer({"X0": cuda.DeviceView(x.ptr, x.shape, x.dtype) if use_view else x, "Y0": np.ones((1, ), dtype=np.float32)})
            assert outputs["identity_out_6"][0] == 2
            assert outputs["identity_out_8"][0] == 2


    def test_subsequent_infers_with_different_input_types(self):
        model = ONNX_MODELS["identity"]
        network_loader = NetworkFromOnnxBytes(model.loader)
        with TrtRunner(EngineFromNetwork(network_loader)) as runner:
            inp = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

            def check(outputs):
                assert np.all(outputs["y"] == inp)

            check(runner.infer({"x": inp}))
            check(runner.infer({"x": cuda.DeviceArray().copy_from(inp)}))
            check(runner.infer({"x": inp}))


    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("7.0"), reason="Unsupported for TRT 6")
    @pytest.mark.parametrize("use_view", [True, False]) # We should be able to use DeviceArray in place of DeviceView
    def test_device_view_dynamic_shapes(self, use_view):
        model = ONNX_MODELS["dynamic_identity"]
        profiles = [
            Profile().add("X", (1, 2, 1, 1), (1, 2, 2, 2), (1, 2, 4, 4)),
        ]
        runner = TrtRunner(EngineFromNetwork(NetworkFromOnnxBytes(model.loader), CreateConfig(profiles=profiles)))
        with runner, cuda.DeviceArray(shape=(1, 2, 3, 3), dtype=np.float32) as arr:
            inp = np.random.random_sample(size=(1, 2, 3, 3)).astype(np.float32)
            arr.copy_from(inp)
            outputs = runner.infer({"X": cuda.DeviceView(arr.ptr, arr.shape, arr.dtype) if use_view else arr})
            assert np.all(outputs["Y"] == inp)
            assert outputs["Y"].shape == (1, 2, 3, 3)


    @pytest.mark.skipif(mod.version(trt.__version__) < mod.version("8.0"), reason="Unsupported before TRT 8")
    def test_cannot_use_device_view_shape_tensor(self):
        model = ONNX_MODELS["empty_tensor_expand"]
        with TrtRunner(EngineFromNetwork(NetworkFromOnnxBytes(model.loader))) as runner, cuda.DeviceArray(shape=(5, ), dtype=np.int32) as arr:
            with pytest.raises(PolygraphyException, match="it must reside in host memory"):
                runner.infer({"data": np.ones((2, 0, 3, 0), dtype=np.float32), "new_shape": arr})
