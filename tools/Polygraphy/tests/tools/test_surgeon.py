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

import onnx
import onnx_graphsurgeon as gs
import pytest
from tests.helper import check_file_non_empty
from tests.models.meta import ONNX_MODELS
from tests.tools.common import run_polygraphy_run, run_polygraphy_surgeon


def onnx_model_sanity_check(model_path):
    run_polygraphy_run([model_path, "--model-type=onnx", "--onnxrt"])


def was_shape_inference_run(status):
    return "ONNX Shape Inference completed successfully" in (status.stdout + status.stderr)


class TestSurgeonExtract(object):
    def test_no_shape_inference_if_has_metadata(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            status = run_polygraphy_surgeon(["extract", ONNX_MODELS["identity_identity"].path, "-o", outmodel.name,
                                             "--inputs", "X:auto:auto"])
            onnx_model_sanity_check(outmodel.name)
            assert not was_shape_inference_run(status)


    def test_onnx_shape_inference_if_no_metadata(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            status = run_polygraphy_surgeon(["extract", ONNX_MODELS["identity_identity"].path, "-o", outmodel.name,
                                             "--inputs", "identity_out_0:auto:auto"])
            onnx_model_sanity_check(outmodel.name)
            assert was_shape_inference_run(status)


    def test_fallback_shape_inference_no_onnx_shape_inference(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            status = run_polygraphy_surgeon(["extract", ONNX_MODELS["identity_identity"].path, "-o", outmodel.name, "--inputs",
                                    "identity_out_0:auto:auto", "--outputs", "identity_out_2:auto", "--force-fallback-shape-inference"])
            onnx_model_sanity_check(outmodel.name)
            assert not was_shape_inference_run(status)


    def test_force_fallback_shape_inference_will_override_model_shapes(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_surgeon(["extract", ONNX_MODELS["dynamic_identity"].path, "-o", outmodel.name, "--outputs", "Y:auto", "--force-fallback-shape-inference"])
            onnx_model_sanity_check(outmodel.name)
            graph = gs.import_onnx(onnx.load(outmodel.name))
            # Inputs should become fixed since fallback shape inference is being forced.
            for tensor in graph.tensors().values():
                assert tensor.shape is not None
            assert tuple(graph.inputs[0].shape) == (1, 2, 1, 1)
            assert tuple(graph.outputs[0].shape) == (1, 2, 1, 1)


    def test_sanity_dim_param(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_surgeon(["extract", ONNX_MODELS["dim_param"].path, "-o", outmodel.name])
            onnx_model_sanity_check(outmodel.name)


class TestSurgeonInsert(object):
    def check_insert_model(self, path, expected_node_ops, expected_graph_input_names, expected_graph_output_names):
        model = onnx.load(path)
        assert [node.op_type for node in model.graph.node] == expected_node_ops

        graph_output_names = set([out.name for out in model.graph.output])
        assert graph_output_names == set(expected_graph_output_names)

        graph_input_names = set([out.name for out in model.graph.input])
        assert graph_input_names == set(expected_graph_input_names)
        return model


    def test_insert_at_tensor(self):
        # Insert a new node in between existing nodes without replacing any existing nodes.
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_surgeon(["insert", ONNX_MODELS["identity_identity"].path, "-o", outmodel.name, "--inputs=identity_out_0",
                                    "--outputs=identity_out_0", "--op=FakeOp"])
            self.check_insert_model(outmodel.name, ["Identity", "FakeOp", "Identity"], ["X"], ["identity_out_2"])


    def test_graph_output(self):
        # FakeOp output tensor should be marked as a graph output. Name should be preserved - identity_out_2
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_surgeon(["insert", ONNX_MODELS["identity_identity"].path, "-o", outmodel.name, "--inputs=identity_out_2",
                                    "--outputs=identity_out_2", "--op=FakeOp"])
            self.check_insert_model(outmodel.name, ["Identity", "Identity", "FakeOp"], ["X"], ["identity_out_2"])


    def test_at_graph_input(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_surgeon(["insert", ONNX_MODELS["identity_identity"].path, "-o", outmodel.name, "--inputs=X",
                                    "--outputs=X", "--op=FakeOp"])
            self.check_insert_model(outmodel.name, ["FakeOp", "Identity", "Identity"], ["X"], ["identity_out_2"])


    # When a specified input tensor is used by multiple other nodes, it should not be
    # disconnected from other nodes.
    def test_multi_use_input(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_surgeon(["insert", ONNX_MODELS["reducable"].path, "-o", outmodel.name, "--inputs=add_out_4",
                                    "--outputs=identity_out_8", "--op=FakeOp"])
            model = self.check_insert_model(outmodel.name, ["Identity", "Identity", "Add", "FakeOp", "Identity"], ["X0", "Y0"], ["identity_out_6", "identity_out_8"])
            other_branch_node = model.graph.node[-1]
            assert other_branch_node.name == "onnx_graphsurgeon_node_7"
            assert other_branch_node.input == ["add_out_4"]


    def test_with_attributes(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            # str_attr='0' should be interpreted as a string, not an int
            # float_attr=0.0 should be interpreted as a float, not an int
            # int_attr=0 should be interpreted as an int
            run_polygraphy_surgeon(["insert", ONNX_MODELS["identity_identity"].path, "-o", outmodel.name, "--inputs=X",
                                    "--outputs=X", "--op=FakeOp",
                                    "--attrs",
                                    "str_attr='0'", "int_attr=0", "float_attr=0.0", "other_str_attr=name",
                                    "str_list_attr=['0','1']", "int_list_attr=[1,2,3]", "float_list_attr=[0.0,-1.0,-2.0]"])
            model = self.check_insert_model(outmodel.name, ["FakeOp", "Identity", "Identity"], ["X"], ["identity_out_2"])
            node = model.graph.node[0]

            attrs = node.attribute
            assert attrs[0].name == "str_attr"
            assert attrs[0].s == b"0"

            assert attrs[1].name == "int_attr"
            assert attrs[1].i == 0

            assert attrs[2].name == "float_attr"
            assert attrs[2].f == 0.0

            assert attrs[3].name == "other_str_attr"
            assert attrs[3].s == b"name"

            assert attrs[4].name == "str_list_attr"
            assert attrs[4].strings == [b"0", b"1"]

            assert attrs[5].name == "int_list_attr"
            assert attrs[5].ints == [1, 2, 3]

            assert attrs[6].name == "float_list_attr"
            assert attrs[6].floats == [0.0, -1.0, -2.0]


class TestSurgeonSanitize(object):
    @pytest.mark.parametrize("fold_shapes", [None, "--no-fold-shapes"])
    @pytest.mark.parametrize("partitioning", [None, "basic", "recursive"])
    def test_fold_constants(self, partitioning, fold_shapes):
        with tempfile.NamedTemporaryFile() as outmodel:
            cmd = ["sanitize", ONNX_MODELS["const_foldable"].path, "-o", outmodel.name, "--fold-constants"]
            if fold_shapes:
                cmd += [fold_shapes]
            if partitioning:
                cmd += ["--partitioning", partitioning]
            run_polygraphy_surgeon(cmd)

            onnx_model_sanity_check(outmodel.name)
            model = onnx.load(outmodel.name)
            assert len(model.graph.node) == 1


    def test_fold_constants_single_pass(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            status = run_polygraphy_surgeon(["sanitize", ONNX_MODELS["const_foldable"].path, "-o", outmodel.name, "--fold-constants", "--num-passes=1"])

            assert "Pass 1" in status.stdout
            assert "Pass 2" not in status.stdout

            onnx_model_sanity_check(outmodel.name)
            model = onnx.load(outmodel.name)
            assert len(model.graph.node) == 1


    @pytest.mark.parametrize("new_dim", [1, 2, 3])
    def test_override_shapes(self, new_dim):
        with tempfile.NamedTemporaryFile() as outmodel:
            cmd = ["sanitize", ONNX_MODELS["dynamic_identity"].path, "-o", outmodel.name, "--override-input-shapes=X:[1,2,{new_dim},{new_dim}]".format(new_dim=new_dim)]
            run_polygraphy_surgeon(cmd)
            onnx_model_sanity_check(outmodel.name)

            model = onnx.load(outmodel.name)

            shape = []
            for dim in model.graph.input[0].type.tensor_type.shape.dim:
                assert isinstance(dim.dim_value, int) and dim.dim_value >= 0
                shape.append(dim.dim_value)

            assert shape == [1, 2, new_dim, new_dim]


    def test_override_shapes_no_clear_const_tensors_meta(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_surgeon(["sanitize", ONNX_MODELS["const_foldable"].path, "-o", outmodel.name, "--override-input-shapes=input:[1,3]"])


    def test_override_shapes_partial_inputs(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_surgeon(["sanitize", ONNX_MODELS["dynamic_identity"].path, "-o", outmodel.name, "--override-input-shapes=Y:[1,2,3,4]"])
            model = onnx.load(outmodel.name)
            assert model.graph.input[0].type.tensor_type.shape.dim[2].dim_param == "height"
            assert model.graph.input[0].type.tensor_type.shape.dim[3].dim_param == "width"


    def test_override_shapes_no_reorder(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            run_polygraphy_surgeon(["sanitize", ONNX_MODELS["reducable"].path, "-o", outmodel.name, "--override-input-shapes", "Y0:[5]", "X0:[5]"])
            model = onnx.load(outmodel.name)
            assert model.graph.input[0].name == "X0"
            assert model.graph.input[1].name == "Y0"


    def test_modify_onnx_outputs(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as outmodel:
            run_polygraphy_surgeon(["sanitize", ONNX_MODELS["identity_identity"].path, "-o", outmodel.name, "--outputs", "mark", "all"])

            model = onnx.load(outmodel.name)
            assert len(model.graph.output) == 2


    def test_cleanup(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as outmodel:
            run_polygraphy_surgeon(["sanitize", ONNX_MODELS["identity_identity"].path, "-o", outmodel.name, "--outputs", "identity_out_0", "--cleanup"])

            model = onnx.load(outmodel.name)
            assert len(model.graph.node) == 1
            assert model.graph.output[0].name == "identity_out_0"


    def test_external_data(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx") as outmodel, tempfile.NamedTemporaryFile() as data:
            model = ONNX_MODELS["ext_weights"]
            assert run_polygraphy_surgeon(["sanitize", model.path, "-o", outmodel.name, "--load-external-data", model.ext_data, "--save-external-data", data.name, "-vvvvv"])
            check_file_non_empty(outmodel.name)
            check_file_non_empty(data.name)


    def test_force_fallback_shape_inference_will_override_model_shapes(self):
        with tempfile.NamedTemporaryFile() as outmodel:
            status = run_polygraphy_surgeon(["sanitize", ONNX_MODELS["dynamic_identity"].path, "-o", outmodel.name, "--force-fallback-shape-inference"])
            onnx_model_sanity_check(outmodel.name)
            graph = gs.import_onnx(onnx.load(outmodel.name))
            # Inputs should become fixed since fallback shape inference is being forced.
            assert tuple(graph.inputs[0].shape) == (1, 2, 1, 1)
            for tensor in graph.tensors().values():
                assert tensor.shape is not None
            assert tuple(graph.outputs[0].shape) == (1, 2, 1, 1)

            assert not was_shape_inference_run(status)
