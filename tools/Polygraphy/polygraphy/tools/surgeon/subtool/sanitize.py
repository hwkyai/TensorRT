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
from polygraphy import mod
from polygraphy.tools import util as tools_util
from polygraphy.tools.args import (DataLoaderArgs, ModelArgs, OnnxLoaderArgs,
                                   OnnxSaveArgs, OnnxShapeInferenceArgs)
from polygraphy.tools.surgeon.subtool.base import BaseSurgeonSubtool

onnx_backend = mod.lazy_import("polygraphy.backend.onnx")
gs = mod.lazy_import("onnx_graphsurgeon")


class Sanitize(BaseSurgeonSubtool):
    """
    Clean up and optimize an ONNX model.
    """
    def __init__(self):
        super().__init__("sanitize")
        self.subscribe_args(ModelArgs(model_required=True, inputs="--override-inputs", model_type="onnx"))
        self.subscribe_args(DataLoaderArgs())
        self.subscribe_args(OnnxShapeInferenceArgs(default=True, enable_force_fallback=True))
        self.subscribe_args(OnnxLoaderArgs(output_prefix=""))
        self.subscribe_args(OnnxSaveArgs(infer_shapes=True, required=True))


    def add_parser_args(self, parser):
        const_fold_args = parser.add_argument_group("Constant Folding", "Options for folding constants")
        const_fold_args.add_argument("--fold-constants", help="Fold constants in the graph by computing subgraphs whose values "
                                     "are not dependent on runtime inputs.", action="store_true", default=None)
        const_fold_args.add_argument("--num-passes", "--num-const-fold-passes", help="The number of constant folding passes to run. "
                                     "Sometimes, subgraphs that compute tensor shapes may not be foldable in a single pass. "
                                     "If not specified, Polygraphy will automatically determine the number of passes required. ",
                                     type=int, default=None, dest="num_const_fold_passes")
        const_fold_args.add_argument("--partitioning", help="Controls how to partition the graph during constant folding: {{"
                                     "'basic': Partition the graph so failures in one part do not affect other parts, "
                                     "'recursive': In addition to partitioning the graph, partition partitions where needed}} ",
                                     choices=["basic", "recursive"], default=None)
        const_fold_args.add_argument("--no-fold-shapes", help="Disable folding Shape nodes and subgraphs that operate on shapes",
                                     dest="fold_shapes", default=True, action="store_false")

        parser.add_argument("--cleanup", help="Run dead layer removal on the graph. This is generally not required if other options are set. ",
                            action="store_true", default=False)
        super().add_parser_args(parser)


    def run_impl(self, args):
        # First do all processing that requires an ONNX-GraphSurgeon graph, then do everything
        # that operates on the ONNX model. This lets us avoid ONNX-GraphSurgeon import if we don't
        # need it.
        def do_graph_processing(model):
            graph = None

            def get_graph():
                nonlocal graph
                if graph is None:
                    graph = gs.import_onnx(model)
                return graph


            user_input_metadata = self.arg_groups[ModelArgs].input_shapes
            if user_input_metadata:
                graph = get_graph()
                graph = tools_util.override_input_shapes(graph, user_input_metadata)

            if self.arg_groups[OnnxShapeInferenceArgs].force_fallback:
                _, layerwise_meta = self.arg_groups[OnnxShapeInferenceArgs].fallback_inference(model)
                graph = get_graph()
                tools_util.set_shapes_from_layerwise_meta(graph, layerwise_meta)

            if args.cleanup:
                graph = get_graph()
                graph.cleanup()

            if graph is not None:
                model = gs.export_onnx(graph)
            return model


        def do_model_processing(model):
            if args.fold_constants:
                model = onnx_backend.fold_constants(model, num_passes=args.num_const_fold_passes,
                                                    do_shape_inference=self.arg_groups[OnnxShapeInferenceArgs].do_shape_inference,
                                                    fold_shapes=args.fold_shapes, partitioning=args.partitioning)
            return model


        model = super().load_model()
        model = do_graph_processing(model)
        model = do_model_processing(model)
        super().save_model(model)
