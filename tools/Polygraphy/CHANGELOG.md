# Polygraphy Change Log

Dates are in YYYY-MM-DD format.


## v0.29.2 (2021-04-30)
### Added
- Added a `--log-file` option to CLI tools to store logging output to a file.
- Added an `--iteration-info` argument to `debug` subtools so that `--check` commands can get information
    about the current iteration.
- Added an experimental `debug repeat` subtool, which is more generic than the existing `debug` subtools.


## v0.29.1 (2021-04-28)
### Changed
- Swapping NumPy arrays to the disk is now disabled by default. It can be re-enabled by setting the
    `POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB` environment variable.


## v0.29.0 (2021-04-28)
### Added
- Added support for per-output `check_error_stat`, which allows different metrics to be checked for different outputs.

### Changed
- Moved JSON utilities into a separate `polygraphy.json` submodule. For backwards compatibility, they remain accessible
    via `polygraphy.util` as well.
- The `max` value for `check_error_stat` in `basic_compare_func` now only checks the maximum absolute/relative tolerances.
    The previous behavior of checking the values element-wise is preserved in the `elemwise` option, which is now the default.

### Fixed
- Fixed a bug where data loader would not cast value ranges provided for `bool` input types, which could lead
    to generating out-of-bound values.


## v0.28.7 (2021-04-26)
### Fixed
- Fixed a bug where NumPy arrays smaller than 8 MiB would be serialized to disk unnecessarily.


## v0.28.6 (2021-04-23)
### Added
- Added a `check_error_stat` option in `basic_compare_func` and corresponding `--check-error-stat` CLI option to control
    which statistic (e.g. mean, median, max) of the error is used to determine whether outputs matched.

### Changed
- A histogram of output/error values will now be displayed at INFO severity on comparison failures.
    Otherwise, it is displayed at VERBOSE severity.

### Fixed
- Fixed a bug where histogram display would wrap to subsequent lines.


## v0.28.5 (2021-04-23)
### Added
- Added more information about absolute/relative difference in `basic_compare_func`. For example,
    it will now print a histogram of the distribution of the outputs and errors.


## v0.28.4 (2021-04-23)
### Added
- Added mean absolute/relative error to `OutputCompareResult`, which is returned by `Comparator.compare_accuracy`.
    This makes it easier to programmatically access this information.

### Changed
- Several improvements to the quality of error messages and warnings.


## v0.28.3 (2021-04-22)
### Fixed
- Fixed a bug where `basic_compare_func` and `DataLoader` would issue warnings when default
    tolerances/value ranges were used.


## v0.28.2 (2021-04-22)
### Fixed
- Fixed a bug where command-line tools would fail if a `--timing-cache` argument was provided
    but the file did not exist.


## v0.28.1 (2021-04-22)
### Fixed
- `basic_compare_func` will now issue warnings if `atol`/`rtol` contain invalid keys.
- `DataLoader` will now issue warnings if `val_range` contains invalid keys.


## v0.28.0 (2021-04-20)
### Added
- Added a `tactic_sources` parameter in `CreateConfig` to control TensorRT's tactic sources.
- Added a `--tactic-sources` argument to CLI tools.
- Added a `DeviceView` class in the `cuda` submodule to represent views of GPU memory.
    `DeviceArray` is now a subclass of `DeviceView`.
- Added support for accepting `DeviceView`s or device pointers in the `Calibrator`. This means that you can now
    run calibration using data already on the GPU.
- Added support for `DeviceView`s in `TrtRunner.infer()`. Note that `DeviceView`s cannot be used for input
    shape-tensors, which must be allocated on the host.
- Added support for using `trt.IInt8Calibrator` as the `BaseClass` of `Calibrator`.
- Exposed some lower level functions like `malloc`, `free`, and `memcpy` in the Polygraphy CUDA wrapper.
- Added a `set_profile()` method to `TrtRunner` to control the active optimization profile.
- Added `-q`/`--quiet` option to CLI tools. This can be used to suppress logging output without eliminating
    all output like `--silent` does.
- Added a `to_trt()` method to `Profile` to convert it to a TensorRT `IOptimizationProfile`.
- Added `--force-fallback-shape-inference` option to `debug reduce`.
- Added `--fail-regex` option to `debug reduce` to distinguish among different types of falures based on command output.

### Changed
- Changed `TRT_LOGGER` to `get_trt_logger()` to make it work properly with lazy imports.
- Further improved lazy imports such that no modules are required in order to import Polygraphy modules.
    Using functionality from Polygraphy modules still requires dependencies.
- Various submodules have been restructured. The old import structure is preserved for backwards compatibility.
- Added `Profile.fill_defaults()` which makes it possible to automatically fill a TensorRT optimization profile
    with sane default values.
- It is now possible to provide TensorRT optimization profile shapes for a subset of the network inputs. In such
    cases, the rest of the profile will be populated automatically with `Profile.fill_defaults()`
- `surgeon extract` will no longer run shape inference unless it is required, e.g. if `auto` is specified for one of
    the shape/data type arguments.
- ONNX shape inference will now be skipped when `--force-fallback-shape-inference` is enabled in `surgeon extract/sanitize`.
- `debug reduce` will now freeze intermediate shapes in the model if `--model-input-shapes` is provided.
- `IterationResult`s now store `LazyNumpyArray` rather than `np.ndarray`.
    The public interface for `IterationResult` will automatically pack or unpack `np.ndarray`s into/from `LazyNumpyArray`,
    so the change is completely transparent.
    This can significantly reduce memory usage for tools like `debug reduce` and `run`.

### Fixed
- Attempting to load a non-existent file will now cause a friendly error message to be displayed rather than crashing.
- `surgeon sanitize` will no longer override shapes other than those specified in `--override-input-shapes`.

### Removed
- Removed optional `symbol` parameter from `lazy_import`.


## v0.27.0 (2021-04-06)
### Changed
- For security reasons, all serialization/deserialization code in Polygraphy has been updated to use JSON
    instead of `pickle`. Use the included `to-json` tool to convert data serialized with `pickle` to JSON
    format.
- Split `TacticReplayer` into separate `TacticRecorder` and `TacticReplayer` classes. This provides more
    fine-grained control over whether to record or replay tactics.
- Deprecated `--tactic-replay` in favor of `--save-tactics` and `--load-tactics`.
- Changed `check_finite` parameter in `Comparator.validate()` to `check_inf`, since it checks whether
    values are non-finite rather than the opposite.

### Fixed
- Polygraphy will now validate command-line arguments so that code-injection is not possible.
- `debug diff-tactics` will now work correctly when replay files are in nested directories.


## v0.26.1 (2021-04-01)
### Added
- Added `--force-fallback-shape-inference` option to `surgeon sanitize` in case ONNX shape inference doesn't work
    well enough to allow for folding.
- Added a `--calibration-base-class` option to allow changing base class for the TensorRT int8 calibrator.

### Fixed
- `FoldConstants` will no longer fail if a constant folding pass fails. Set `error_ok=False` to disable this behavior.


## v0.26.0 (2021-03-30)
### Added
- Added support for saving/loading ONNX models with externally stored weights.
- Added support for automatically installing dependencies as they are needed. This behavior can be enabled by
    setting the `POLYGRAPHY_AUTOINSTALL_DEPS` environment variable to `1`.
    When auto-install is enabled, Polygraphy can also automatically upgrade existing packages if a newer version is requested.
- Added `error_ok` option in `InferShapes`, which can be set to `False` to make the loader raise an error when shape
    inference fails.

### Changed
- `val_range` in the DataLoader now falls back to the default range if no range is specified for an input.
- `atol` and `rtol` in `CompareFunc.basic_compare_func` now fall back to the default tolerance values
    if no tolerance is specified for an output.
- Folding shapes is now optional in `FoldConstants`.
- `surgeon sanitize` now includes a `--no-fold-shapes` option to disable shape folding.

### Fixed
- Fixed a bug in `surgeon insert` where input tensors would be disconnected from all their consumers.
    Previously, in a model with branches, if one entire branch was replaced by `surgeon insert`, the other branch
    would be invalidated. This is no longer the case.
- `run` will now attempt to avoid introducing a dependency on the `onnx` Python package when using an ONNX
    model when `--trt` is the only specified runner.
- When `--force-fallback-shape-inference` is set in `surgeon extract`, it will now correctly ignore shapes
    already inferred in the model.
- ONNX loaders will no longer make a copy of the model unnecessarily. If a copy is desired, the `copy` parameter can
    be set to `True` for loaders that may modify the model.
- `InferShapes`/`infer_shapes` will now work with ONNX models larger than 2 GiB if a path to the model is provided
    instead of an `onnx.ModelProto`
- Fixed a bug where `FoldConstants` would not count nodes within subgraphs.

### Removed
- Removed `OnnxTfRunner` and associated CLI options.


## v0.25.1 (2021-03-15)
### Added
- Added `--partitioning` flag to `surgeon sanitize` to control how ONNX-GraphSurgeon partitions the graph during
    constant folding.
- Added `--cleanup` flag to `surgeon sanitize` to remove dead layers in ONNX models.

### Changed
- `ExtractSubgraph` loader will now fallback to using shapes/dtypes defined in the model when none are specified.

### Fixed
- `surgeon sanitize` no longer runs inference when the `--override-input-shapes` option is set. Instead, intermediate shapes are cleared.
- `surgeon extract` will no longer override shapes or data types already set in the model when running fallback shape inference.


## v0.25.0 (2021-03-01)
### Added
- Added support for list attributes in `surgeon insert`.
- Added `val_range` parameter to data loader, which is more generic than `int_range`/`float_range`, which are now deprecated.
- Added support for per-input data ranges to `val_range` parameter.
- Added `--val-range` CLI option to set input ranges on the command-line.
- Added `:` as a valid separator for various options and `[dim0,...,dimN]` as valid syntax for shapes.
    For example, you can now optionally use:
    ```bash
    --inputs input0:[3,4]:int64 input1:[4,64,64]:float32
    ```
    instead of:
    ```bash
    --inputs input0,3x4,int64 input1,4x64x64,float32
    ```
    The new and old styles cannot be mixed.
- Added support for specifying per-output top-k values to CLI tools.
- Added `--trt-config-script` argument, which allows CLI tools to accept scripts that define functions that create TensorRT
    builder configurations.
- Added `--data-loader-script` argument, which allows CLI tools to accept scripts that define data loaders.
- Added a new example for the `convert` CLI tool, which shows how to use a custom data loader for int8 calibration on the command-line.

### Fixed
- Fixed a bug where `debug reduce` would remove branches even if they were required to reproduce failures.


## v0.24.2 (2021-02-25)
### Added
- Added support for string input types in `OnnxrtRunner`.
- Added `reduce` subtool to `debug` which can reduce failing ONNX models to the smallest possible failing subgraph.

### Fixed
- ONNX loaders will no longer modify the original model provided, but instead make a shallow copy.


## v0.24.1 (2021-02-22)
### Added
- Added an example to `dev/` showing how to write new command-line tools.

### Fixed
- Verbose TensorRT network logging will no longer fail to show attributes for layers on older versions of TensorRT.


## v0.24.0 (2021-02-19)
### Added
- `convert` can now automatically determine the output model type based on the file extension.
- Added immediately evaluated functional variants for all loaders exported by Polygraphy.
    The variants use the same name as the loaders, except `snake_case` instead of `PascalCase`.
    See [the example](./examples/api/06_immediate_eval_api) for details.

### Changed
- Polygraphy no longer has `numpy` as an install requirement.
    Note however that most, but not all, APIs and CLI tools in Polygraphy still do require `numpy`.

### Removed
- Removed `func.invoke()` since immediately evaluated functions now supersede it.

### Fixed
- Fixed a bug where some `debug` subtools would write engines to the wrong path.


## v0.23.4 (2021-02-15)
### Added
- Added `FoldConstants` loader for ONNX models.
- Added `ExtractSubgraph` loader for ONNX models.

### Changed
- Moved `--fp-to-fp16` option to `convert`.


## v0.23.3 (2021-02-13)
### Added
- Added `ConvertToFp16` as a separate loader for ONNX models.
- Added `InferShapes` loader for ONNX models.

### Changed
- `surgeon sanitize` will now run shape inference by default.
- `Modify<X>` loaders have been renamed to `Modify<X>Outputs` to better reflect their purpose.
- `surgeon sanitize` can now run multiple passes of constant folding to deal with nodes that may not be folded
    after the first pass (for example, `Shape` nodes in cases where ONNX shape inference does not complete).


## v0.23.2 (2021-02-11)
### Added
- Added an experimental `debug` subtool, which includes `build` and `diff-tactics` (formerly part of `flaky`)
    and `precision` (formerly a separate tool).

### Changed
- `flaky diff-tactics` will now only show layers that have potentially bad tactics.
    To view an entire tactic replay, use `inspect tactics`
- `flaky repeat` will now only log command `stderr` output with `ERROR` severity if the command failed.
    Otherwise, `stderr` output is logged with `WARNING` severity.
- `TacticReplayer` can now accept a `TacticReplayData` instance directly.
- `TacticReplayData` can now be constructed manually instead of relying on TensorRT types.

### Removed
- `flaky` and `precision` tools have been removed and replaced by the `debug` subtool, which includes the
    functionality of both.


## v0.23.1 (2021-02-05)
### Added
- Added a `POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS` environment variable to enable internal correctness checks at runtime.
    By default, these checks are disabled. A failure in such a check typically indicates a bug in Polygraphy.
- Added context managers for CUDA helper classes. This helps ensure they are correctly freed.
- Added `sparse_weights` parameter to `CreateConfig`, which enables TensorRT optimizations related to sparse weights.
- Added a `--sparse-weights` option to various CLI tools.

### Fixed
- Added checks for cases where paths provided to `BytesFromPath` did not exist.


## v0.23.0 (2021-02-02)
### Added
- Added `__enter__`/`__exit__` to `Calibrator` so that device buffers can be reliably freed after calibration using a context manager.
- Added `fp_to_fp16` parameter to `ModifyOutputs` which will use `onnxmltools` to convert float tensors in the model to 16-bit floats.
- Added `--fp-to-fp16` CLI argument to various tools.
- Added support for `float`, `int`, and `str` attributes to `surgeon insert`.
- Added `InvokeFromScript` loader, which can import and invoke a function from a Python script.
- Added support for loading TensorRT networks from Python scripts to various CLI tools.
    CLI tools can now accept a Python script in place of a model file.
    The script should define a `load_network` function that takes no arguments and returns a TensorRT builder, network, and optionally parser.
    See [the example](examples/cli/run/04_defining_a_trt_network_manually/) for details.
- Added an experimental `template` tool that can generate template files.
    - Added a `trt-network` subtool that can generate a template script for defining TensorRT networks using the network API.
- Added a `SaveBytes` loader to facilitate writing bytes to a file between loaders.
- Added an experimental `flaky` tool that can help debug flaky failures.
    - Added `repeat` subtool, which will run a command repeatedly and sort artifacts into `good` and `bad` directories.
    - Added `diff-tactics` subtool, which compares known-good and known-bad tactic replay files to determine which tactics may be the source of error.

### Changed
- `EngineFromNetwork` and `CreateConfig` no longer use the global timing cache by default.
- Changed `--timing-cache` default in CLI tools to `None`.
- Changed `timing_cache` parameter to `load_timing_cache` and `save_timing_cache` in `CreateConfig` and `EngineFromNetwork` respectively.
- Runners will now raise errors in `infer` if the provided input data types or shapes do not match expected types and shapes.
    This behavior can be disabled by setting `check_inputs=False`.
- Changed `--toposort` default to off in `surgeon` tools as ONNX models are typically topologically sorted.
- The logger will now log messages with `WARNING` or greater severity to `sys.stderr` instead of `sys.stdout`

### Removed
- Removed `CNTKRunner` and `--cntk` CLI option.
- Removed experimental `--network-api` flag in CLI tools. This is superseded by the `template trt-network` subtool.

### Fixed
- Fixed memory leaks in `EngineFromNetwork`, `EngineFromBytes`, and `TrtRunner`.


## v0.22.0 (2021-01-20)

### Added
- Added support for timing caches in `EngineFromNetwork` and `CreateConfig`.
    The former can generate caches, while the latter can load them, resulting in much faster engine builds.
    By default, Polygraphy will use a global timing cache in the temporary directory.
- Added a `--timing-cache` option to various CLI tools.
- Added an `EngineBytesFromNetwork` TensorRT loader to provide serialized engines directly.
- Added a `BytesFromEngine` TensorRT loader to provide a means of in-memory engine serialization.
- Added an experimental `convert` subtool, which can convert models to various other formats.
- Added an `algorithm_selector` parameter to `CreateConfig` to allow the user to override TensorRT's tactic choices.
- Added a `TacticReplayer` algorithm selector to allow for recording and replaying tactics in the TensorRT builder.
    This makes it possible to make the TensorRT builder behave deterministically.
- Added an experimental `--tactic-replay` option to various CLI tools to make it possible to record to and replay from tactic replay files.
- Added an experimental `inspect` subtool, `tactics` which can display tactic replay files in a human readable format.

### Changed
- The `surgeon sanitize` subtool can now also modify model outputs.
- `surgeon insert` will now preserve graph input and output names.

### Fixed
- Fixed a bug where the CUDA wrapper could not allocate buffers larger than 3GiB.


## v0.21.1 (2021-01-12)

### Added
- `TrtRunner` can now optionally accept a `context` directly instead of an `engine`.
- `basic_compare_func` will now show mismatched indices in addition to mismatched values.


## v0.21.0 (2020-11-30)
### Added
- Added an experimental `surgeon` subtool, `insert`, which can insert new nodes into an ONNX model.
- Added an experimental `surgeon` subtool, `sanitize`, which can remove unused nodes and fold constants in an ONNX model.
- Added `--load-inputs` and `--save-inputs` to provide a mechanism to supply custom input data on the command line.
- Added `func.invoke()`, a function that calls a provided callable.
    This can be useful to make it more obvious that a loader is being immediately evaluated.
    For example: `EngineFromNetwork(...)()` vs. `func.invoke(EngineFromNetwork(...))`
- Added per-output tolerance support in `basic_compare_func`.
- Added per-output tolerance support to the `--atol` and `--rtol` command-line options.

### Changed
- Renamed `inspect results` to `inspect data` since it can now also be used to inspect input data, not just results.
- `Comparator.compare_accuracy` now supports comparing a single runner against itself.

### Removed
- Removed experimental surgeon subtool `prepare` and `operate` as they were difficult to maintain and not very useful.

### Fixed
- Fixed a memory leak due to `IBuilderConfig` not being properly freed in the `EngineFromNetwork` loader.
- Fixed memory leaks on exceptions in TensorRT loaders.
- Fixed a bug in `inspect model` where `dim_param`s in ONNX models would show up as `-1`.


## v0.20.13 (2020-10-08)
### Added
- Shape values in `TensorMetadata` can now be strings to indicate dynamic dimensions.
- `TRT_LOGGER` is now exported under `polygraphy.backend.trt`

### Fixed
- Fixed a bug in `surgeon extract` where ONNX models using `dim_param` would be rejected.


## v0.20.12 (2020-10-01)
### Added
- Added missing copyright headers


## v0.20.11 (2020-09-25)
### Added
- Added an `--input-shapes` alias for the `--inputs` option in `run` to better reflect its purpose.

### Fixed
- `inspect model` will no longer show `dtype`/`shape` as `None` if the information is not present in the model. Instead, these are now omitted.


## v0.20.10 (2020-09-23)
### Fixed
- Fixed a bug where boolean outputs would cause a crash in `basic_compare_func`


## v0.20.9 (2020-09-22)
### Fixed
- Fixed a bug where `TrtRunner` would use the wrong shapes for empty tensor outputs .


## v0.20.8 (2020-09-22)
### Fixed
- Fixed a bug where the `Calibrator` would not re-check the cache when `reset()`


## v0.20.7 (2020-09-22)
### Added
- Added `-v`/`--version` flag to `polygraphy`

### Changed
- Cleaned up unnecessary logging output, and fixed formatting.


## v0.20.6 (2020-09-18)
### Added
- Added new modes to `inspect model`, to control whether to show weights in the model.
- Added `-s`/`--show-values` option to `inspect results` to display output values.
- Added an experimental `--top-k` flag to `run`, which will apply a Top-K before comparing outputs.
- Added `exclude_outputs` to `ModifyOutputs` and `ModifyNetworkOutputs`
- Added an experimental `--onnx-exclude-outputs` and `--trt-exclude-outputs` to selectively unmark outputs.


## v0.20.5 (2020-09-16)
### Fixed
- Fixed a bug in `inspect model` for ONNX models containing nodes with Tensor attributes.
- Fixed a bug where `DeviceArray.copy_from` would segfault in rare cases.


## v0.20.4 (2020-09-14)
### Fixed
- General cleanup and addition of missing docstrings.


## v0.20.3 (2020-09-11)
### Fixed
- Fixed a bug where `DataLoader` would use a shape provided by the user even for static shapes in the model.
- Fixed a bug where `DataLoader` would incorrectly report certain tensors as shape tensors.
- Fixed a bug where the `DataLoaderCache` would stop checking the cache after the first miss.


## v0.20.2 (2020-09-11)
### Added
- Added an `extend` decorator, which makes it easier to extend existing loaders.
- Added more API examples.
- `Comparator.compare_accuracy` will now display an accuracy summary after processing all iterations.
- Added a `CreateNetwork` loader to create new TensorRT networks
- Added experimental `--network-api` option that works with `--gen` to allow manually defining a TensorRT network.

### Changed
- `Calibrator` can now accept a file-like object for `cache` instead of just a file path.

### Fixed
- Fixed various errors in API documentation.
- `EngineFromBytes` will now call `trt.init_libnvinfer_plugins` before attempting to deserialize the engine.


## v0.20.1 (2020-09-09)
### Added
- Added HTML docs for the Python API

### Fixed
- Fixed a bug where the data loader would not support cases where `int_min` == `int_max` when bounding input data
- Fixed a bug where OnnxrtRunner would report incorrect metadata for ONNX models using `dim_param` for dynamic dimensions.


## v0.20.0 (2020-09-08)
### Added
- `CreateConfig` now accepts a `strict_types` argument.
- Added a new `polygraphy` binary, which includes several tools
- Added an experimental new tool: `precision`, which can be used to figure out what layers to run in higher precision in TensorRT to achieve the desired accuracy.
    - Added `bisect` subtool that does binary search
    - Added `linear` subtool that does a linear search
    - Added `worst-first` subtool that marks the layers that introduce the most error first.
- Added a new tool: `inspect` to inspect supported files
    - Added `model` which displays information about models.
    - Added `results` which displays information about saved `RunResults`
- Added back `subprocess_polling_interval` to `Comparator.run()`, as this is still required in certain rare cases.
- Optimization passes are now optional in `OnnxFromTfGraph`, and can be disabled by setting `optimize=False` in the constructor.
- Runners now include an `is_active` property, which indicates whether the runner is currently activated.
- Added an experimental new tool: `surgeon`, which can be used to modify ONNX models more easily than using ONNX-GraphSurgeon.
    - Added `prepare` and `operate` which can be used to modify an ONNX model using a JSON configuration.
    - Added `extract` which can extract ONNX subgraphs with a single command.
- Added `--onnx-outputs` and `--trt-outputs` to set outputs in the corresponding loaders
- Added a passthrough loader, `LoadPlugins`, that can wrap any other loader, and load plugins

### Changed
- `EngineFromNetwork` will no longer free the the builder, network and parser if they are provided directly (as opposed to via a loader).
- `TrtRunner` will no longer free the the engine if it is provided directly (as opposed to via a loader).
- All file saving arguments now take file paths instead of directories. This makes it easier to know exactly where each file is being written.
- `compare_func` in `Comparator.compare_accuracy` now accepts a function that returns anything convertible to a boolean, rather than requiring a boolean.
- `basic_compare_func` now will return information about required tolerances after `Comparator.compare_accuracy`.
- `Calibrator` can now be configured to inherit from a different TensorRT calibrator base class.
- ONNX GraphSurgeon is no longer required to mark outputs in ONNX models.
- `TrtLegacyRunner` no longer depends on `pycuda`
- `TrtRunner` will now only reset context shapes if the shapes changed. This should improve performance.
- `DataLoader` now takes `int_range` and `float_range` parameters, so min/max can be provided more concisely.
- All `Loaders` and `Runner` were renamed to better reflect their purpose, and to improve readability.
- Renamed `warm_up_runs` to `warm_up`.
- `Calibrator`'s `data_loader` parameter now accepts any generator or iterable instead of requiring a special type.
- `Comparator.run`'s `data_loader` parameter now accepts any generator or iterable instead of requiring a special type.
- The included `DataLoader` can now be used as an iterable, and its iteration length can be controlled via the `iterations` parameter.
- Renamed `--input-shape` to `--inputs`
- Renamed `--min-shape`/`--opt-shape`/`--max-shape` to `--trt-min-shapes`/`--trt-opt-shapes`/`--trt-max-shapes`
- `DataLoader` now accepts an `input_metadata` parameter which can be used to override shapes and data types.
- Split off `layerwise` and `outputs` functionality into separate `Modify` loaders.
- Split off artifact saving functionality into separate `Save` loaders.
- Renamed `--read` options to `--load`, and `--write` to `--save`
- Renamed `--read-outputs`/`--write-outputs` to `--load-results`/`--save-results`
- `Calibrator` no longer requires `input_metadata` to be set if the data loader does not need it
- `TfRunner` now uses a `CreateConfig` loader to supply configuration.
- `TfRunner` and `OnnxrtRunner` now take a `BuildSession`, so that custom sessions can be used.

### Removed
- Removed iteration arguments from `Comparator.run()` and `Calibrator`. Instead these now iterate the provided data loader until it runs out of data.
- Removed `--load-engine` option from `polygraphy`. Engines can now be provided as models directly, e.g. `polygraphy run example.engine --trt`
- `polygraphy_exec` and `polygraphy_gen` were removed. They are superseded by the `run` subtool of `polygraphy`.
- `--layerwise` and `layerwise` options have been removed. Layerwise behavior is now possible with `outputs=constants.MARK_ALL` or `--<framework>-outputs mark all`

### Fixed
- Fixed bugs in `Comparator.validate` that would cause it not to correctly display non-finite values.
- `Calibrator` will now warn if a cache exists but is empty
- `DataLoader` will now used a fixed seed value unless otherwise specified. This ensures consistent run-to-run behavior.
- The default `find_output_func` will no longer compare outputs whose names don't match if there is another output that does match.
- Fixed a bug where custom names provided to runners would still be suffixed with a timestamp.
- Fixed a bug where regular TensorRT calibrators could not be used with `CreateConfig`
- The missing subtool warning will no longer be displayed if that subtool is not being used.


## v0.17.0 (2020-07-20)
### Added
- `basic_compare_func` now accepts a `find_output_func` parameter, allowing users to control which outputs are compared between results.
- The `--load-outputs` argument can now accept multiple different files. Outputs from each of these will be read in order.
- Added an implicit batch ONNX network loader for the legacy TensorRT runner. This will not work with recent versions of the parser.
- Added `RunResults` class which replaces the `OrderedDict` that `Comparator.run` previously returned (structure is unchanged).

### Changed
- `layerwise` mode will no longer mark constants as outputs.
- The default `compare_func` in `Comparator.compare_accuracy` will now always iterate over the output names in the first `IterationResult` and attempt to find them in the second. The order of the `IterationResult`s provided to this function can be modified either by setting `comparisons` in `Comparator.compare_accuracy`, or changing the order of runners in `Comparator.run`
- Improves `polygraphy_gen` output formatting
- Renamed `RunResult` to `IterationResult` to better reflect its purpose.
- Default runner names now include timestamps to disambiguate when saving and loading multiple runners.
- `graphsurgeon` is no longer a dependency of Polygraphy

### Fixed
- Logger settings in `polygraphy_exec`/`polygraphy_gen` are now set prior to any logging output.
- Comparator will no longer attempt to decompress all `bytes` objects sent over the queue when using subprocesses


## v0.16.0 (2020-06-11)
### Added
- Added `OnnxExtWeightsNetworkLoader` to support loading ONNX models with externally stored weights into TensorRT.
- Added a `TensorMetadata` class to replace dictionaries that were used across Polygraphy.
- Added `CaffeNetworkLoader` for the `TrtLegacyRunner`

### Changed
- `polygraphy_exec` and `polygraphy_gen` will no longer use subprocesses by default. To revert to the old behavior, the `--use-subprocess` flag must now be explicitly provided.
- `SerializedEngineLoader` now accepts a `buffer_loader`, so that a function that loads a serialized engine may be provided instead of the serialized engine itself.
- Default opset for `OnnxFromTfGraph` has been updated to `11`

### Fixed
- `polygraphy_exec` and `polygraphy_gen` now correctly handle cases where no model file is provided


## v0.15.0 (2020-05-05)
### Added
- Added a `PolygraphyException` class to serve as a base class for exceptions raised by Polygraphy.

### Changed
- `ConfigLoader` now accepts a list of `Profile`s to support multiple optimization profiles.
- Changed the format of CLI shapes arguments to enable specifying multiple profiles.
- Moves `outputs` argument from TfRunner to the tensorflow loaders.


## v0.14.1 (2020-04-17)
### Added
- Polygraphy now includes a thin `ctypes` wrapper around the CUDA runtime library, accessible in `util/cuda.py`

### Changed
- `TrtRunner` no longer depends on `pycuda`, and instead uses the included CUDA wrapper.
- Loader parameters may now be loaders themselves, or the result of invoking a loader.
- Improves the quality of Comparator messages when there are mismatches
- `basic_compare_func` will now preserve output ordering in the results.
- Makes `EngineFromNetwork` compatible with TensorRT 7.0


## v0.14.0 (2020-04-09)
### Added
- Restructures ONNX Runner, and adds layerwise functionality (using ONNX-GraphSurgeon).
- Added `--timestamp` and `--line-info` options to `polygraphy_exec` to enable logging of timestamp and line numbers respectively.
- Added `--no-letter` option to disable severity letter prefixes in log messages
- Added `register_callback` to Logger, which registers a callback that will be called whenever the severity changes.
- Added `Logger.verbosity()` which returns a context manager that can be used to temporarily change logging severity.
- Added new variants to `--model-type` in `polygraphy_exec`: `keras`, `ckpt`, renamed `tf` to `frozen`
- Added `ConfigLoader` which can be passed to `EngineFromNetwork` to customize the build configuration prior to building.

### Changed
- The logger no longer displays timestamps and line numbers. These can be enabled by setting the `timestamp`/`line_info` properties respectively to `True`.
- Logger now relies on the `colored` module to provide colored output
- `polygraphy_exec` now runs runners in the order in which they were specified.
- Greatly shortens import paths by removing `_runner` suffixes and shortening framework names (e.g. `tensorflow_runner` -> `tf`)
- `runners` submodule has been renamed to `backend`
- `TrtRunner` has been renamed to `TrtLegacyRunner`
- `TrtRunnerV2` has been renamed to `TrtRunner`
- `polygraphy_gen` is now at parity with `polygraphy_exec`

### Removed
- Removed `--tftrt` as a separate runner in `polygraphy_exec` - instead it is now an option for the `--tf` runner.
- Removed `--tftrt-gpu-memory-fraction` and renamed `--tf-gpu-memory-fraction` to `--gpu-memory-fraction` in `polygraphy_exec`
- Removed `--tfonnx`, and instead adds this functionality in `--onnxrt` when using a TensorFlow model in `polygraphy_exec`
- Removed `Experimental` argument section in `polygraphy_exec`. All functionality has now been integrated into non-experimental arguments.
- Removed `preprocess_network` argument from `EngineFromNetwork`. This functionality can be achieved by wrapping the network loaders instead.


## v0.13.4 (2020-03-25)
### Changed
- `Comparator.run` will now forcefully terminate the subprocess if it does not exit on its own.


## v0.13.3 (2020-03-20)
### Added
- Added TF32 support to legacy TrtLegacyRunner.


## v0.13.2 (2020-03-20)
### Changed
- Various improvements to automatic shape matching for cases where shapes between runners do not match exactly.
- Changed `BaseRunner` so that runners can now implement `activate()`/`deactivate()` instead of `__enter__()`/`__exit__()`
- `polygraphy_exec` now defaults to running just a single iteration of inference.

### Removed
- The `--accuracy` flag has been removed from `polygraphy_exec`, as this is now the default behavior.

### Fixed
- TensorRT runners now use the same builder to build the network and engine, instead of using a separate builder for each.


## v0.13.1 (2020-03-17)
### Fixed
- Fixes a bug in `try_match_shape`


## v0.13.0 (2020-03-17)
### Added
- Added a `tf32` parameter as well as `--tf32` flag for TensorRT.
- Added support for `dim_param` in ONNX.

### Changed
- `fp16_mode` and `int8_mode` parameters have been renamed to `fp16` and `int8` respectively.
- `polygraphy_exec` will now use the runtime shapes specified rather than always using `OPT` shapes from the TensorRT profile.
- Improves shape matching logic in `DataLoaderCache`


## v0.12.0 (2020-03-06)
### Added
- Added a `start_index` and `end_index` to `Comparator.run` to make it easy to skip over inputs from the data loader.
- Added `CompareFunc` to provide built-in comparison functions.
- Added `PostprocessFunc` to provide built-in post-processing functions.
- `Comparator.compare_accuracy` now returns an `AccuracyResult` object, which contains much more information about the results of the comparisons.
- Added `percentage()` function to `AccuracyResult` to provide an easy way to figure out the percentage of passed iterations.

### Changed
- Replaces `RunInfo` with `IterationResult`. The latter only stores information about a single iteration for a single runner.
- `compare_func` in `Comparator.compare_accuracy` is now a `Callable(IterationResult, IterationResult) -> Dict[str, bool]`
- `warm_up_runs` now defaults to `0`, and `end_index` to `1`
- Ordering of outputs in a single iteration is now preserved in `CompareFunc.basic_compare_func`
- `use_subprocess` now defaults to `False` in `Comparator.run()` (still defaults to `True` in `polygraphy_exec`).
- `Calibrator` now takes a `start_index` and `end_index` argument instead of `max_items`.

### Removed
- Removed `Comparator.compare` function since `Comparator.compare_accuracy` includes all of its functionality.
- `iterations` in `Comparator.run` has been removed and replaced by `start_index` and `end_index`
- Removed `subprocess_polling_interval` argument, as `Comparator` can now properly detect when the subprocess terminates.

### Fixed
- `Comparator.run()` will no longer hang if there is a segfault in the subprocess.


## v0.11.3 (2020-02-25)
### Added
- Added `--int-min`, `--int-max`, `--float-min`, and `--float-max` arguments to `polygraphy_exec`
- Added `--explicit-precision` option to `polygraphy_exec` to enable QAT models in TRT.
- Added empty tensor support. Empty tensors are tensors whose shapes contain one or more 0s.

### Changed
- When `--load-outputs` or `--save-outputs` is specified to `polygraphy_exec`, `seed` will default to `1` to ensure consistent inputs across runs.


## v0.11.2 (2020-02-11)
### Added
- Added a `--calibration-cache` option to `polygraphy_exec` to enable supplying a calibration cache
- Added a `--no-color` option to disable color logging.


## v0.11.1 (2020-02-11)
### Added
- Added `GraphOptimizerLoader` for freezing TensorFlow graphs and `--freeze-graph` option to `polygraphy_exec`.
- Added `--load-outputs` and `--save-outputs` to `polygraphy_exec` for comparing across executions.
- Added `KerasLoader` for loading models stored in `hdf5` format.
- Added constant folding pass to `GraphOptimizerLoader` for TensorFlow graphs.

### Changed
- Updates `Calibrator` so that it will now use the opt dimension of a profile for networks with dynamic shapes.
- Updates Legacy TensorRT runner to use `Loaders` for easier UFF debugging.

### Fixed
- `Calibrator` will no longer allocate buffers if a calibration cache was provided.


## v0.11.0 (2020-01-28)
### Added
- Added generation of ONNX code to `polygraphy_gen`
- Added default implementations of some `BaseRunner` methods.
- Added `last_inference_time()` to `BaseRunner` so that `infer()` now only needs to return outputs.
- Added `Calibrator` for int8 calibration, along with additional parameters to `EngineFromNetwork`

### Changed
- Better warnings for user-defined implementations of various APIs.
- `DataLoaderCache` will now warn loudly when a set of inputs needs to be regenerated.
- Cleans up `Comparator` `run()` function.
- Moves most `save_*` options into loaders rather than runners.
- Changed `BaseDataLoader.next()` to take index as an argument. This way, inputs can be reliably repeated across runners.
- Moves all `layerwise` parameters into loaders rather than runners.
- `Loader`s are now interchangeable with Python `Callable`s
- `DataLoader`s are now interchangeable with Python `Callable`s

### Fixed
- `DataLoader` no longer generates all `True` values for boolean types.
- Various bug fixes in `polygraphy_gen`
- `DataLoaderCache` is now sent over the queue when runners are run in subprocesses. This resolves an issue where the cache was not being updated correctly.
- `Comparator` now updates runners correctly when using a subprocess.


## v0.10.6 (2019-12-11)
### Added
- Added `--no-fold-constant` option to prevent `OnnxFromTfGraph` from doing constant folding in the TensorFlow graph.
- Added experimental `polygraphy_gen` script that enables generation of template Python scripts for running Polygraphy.

### Fixed
- Bug fix for cases where TensorFlow nodes with no outputs are recognized as graph outputs by `GraphSurgeon`.


## v0.10.5 (2019-12-9)
### Added
- Added `name` parameter to `CheckpointLoader` in case the checkpoint does not include a `checkpoint` file.

### Changed
- `TFTRTLoader` now accepts any kind of TensorFlow Graph loader

### Fixed
- Bug fix in `TrtRunner` `Buffers` so that no-op reshapes (no reallocation) are handled correctly.


## v0.10.4 (2019-12-4)
### Added
- Added `check_inf`, `check_nan`, and `fail_fast` options to `Comparator.validate()`

### Changed
- Cleans up `Buffers` implementation for `TrtRunner` - eliminates an unnecessary copy that was happening on the host input.
- Improved logic for matching output names in `util.find_in_dict()`


## v0.10.3 (2019-11-18)
### Changed
- `TrtRunner` will no longer call `context`'s shape setting functions on non-dynamic inputs.

### Fixed
- Bug fix for volume computation for scalars.
- Updates `DataLoader` to handle scalars correctly, adds several tests.


## v0.10.2 (2019-11-11)
### Added
- Added various utility functions as static members of `TrtRunner`, e.g. `create_network` function to simplify TensorRT's network flags.

### Changed
- `EngineFromNetwork` will now mark network outputs when `layerwise=True`


## v0.10.1 (2019-10-31)
### Added
- Added support for `bool` outputs in `Comparator`


## v0.10.0 (2019-10-28)
### Changed
- Replaces `OnnxEngineLoader` with `OnnxNetworkLoader` and `EngineFromNetwork`. This allows for more flexibility in building engines from TensorRT networks.


## v0.9.8 (2019-10-24)
### Added
- Added `allow_growth` option to TfRunner to work around `CUDNN_STATUS_INTERNAL_ERROR`. When `allow_growth` is enabled, the error disappears.

### Changed
- `DataLoaderCache` will now attempt to permute inputs in cases where shapes do not match exactly (e.g. NCHW vs NHWC inputs).

### Fixed
- Fixes a bug in `polygraphy_exec` which caused it to ignore user-defined profiles.


## v0.9.7 (2019-10-18)
### Added
- Added support for many more ONNX data types.
- Added support for `int8` and explicit precision mode in `TrtRunner`
- Added `preprocess_network` parameter to `OnnxEngineLoader` so that the network can be modified before it is used for building.

### Changed
- `TrtRunner` will now attempt to generate sane default shapes in cases with dynamic shapes where no profiles are provided.


## v0.9.6 (2019-10-15)
### Changed
- `DataLoader` no longer overrides static shapes in the model, but issues a warning if an override is requested.
- `DataLoader` now accepts shape tensor inputs in its `default_shapes` parameter.


## v0.9.5 (2019-10-9)
### Added
- Added timestamps to logging output.

### Fixed
- `Comparator` can now catch segfaults in runners properly.


## v0.9.4 (2019-10-7)
### Added
- Added options for `DataLoader` to be able to specify input bounds
- Added smarter matching for input metadata in the `DataLoaderCache`

### Changed
- Default `subprocess_polling_interval` is now 30 seconds.
- `Comparator` now attempts to partially match output names when no exact matches are found.


## v0.9.3 (2019-10-1)
### Added
- Added `subprocess_timeout` parameter to `Comparator.run` to prevent hangs when a subprocess does not terminate.
- Added `subprocess_polling_interval` parameter to allow the process to be polled so that failing processes can be terminated before the full `subprocess_timeout`.


## v0.9.2 (2019-10-1)
### Changed
- If ONNX checker fails due to the IR version of the model being too new, Polygraphy now ignores the error and continues.


## v0.9.1 (2019-10-1)
### Changed
- `OnnxEngineLoader` now accepts an `onnx_loader` for better flexibility in loading models.
- `polygraphy_exec` now supports running TF models in TRT via the tf2onnx converter.
- Legacy `TrtLegacyRunner` now only supports UFF models.


## v0.9.0 (2019-09-30)
### Added
- Added `BaseModelLoader` that can be used to load models. This allows for reuse of existing runners with different import paths. For example, `OnnxrtRunner` can be used with `OnnxFromTfGraph` in order to run a TensorFlow frozen graph via ONNX Runtime.
- Implements `ModelLoader`s for `TfRunner`, including a frozen model loader, checkpoint loader, and TF-TRT loader.

### Changed
- `OnnxFromTfGraph` now accepts a TensorFlow ModelLoader to support a wider variety of input formats.
- Updates legacy `TrtLegacyRunner` to use `get_input_metadata` API, so it is usable for UFF models.


## v0.8.1 (2019-09-26)
### Changed
- Comparator will now look at the union of all outputs from all runners when checking for common outputs.
- `TrtRunner` will no longer mark layers within the loop body as network outputs in `layerwise` mode.
- `DataLoaderCache` now falls back to reusing inputs based on order if names do not match exactly.
- `DataLoader` now accepts a `default_shapes` parameter to override dynamic shapes.


## v0.8.0 (2019-09-18)
### Added
- Added `get_input_metadata` API to BaseRunner. Overhauls runners so they no longer need to handle dynamic input shapes individually.
- Added `DataLoader` class which can be used to feed data to the Comparator.
- Added `DataLoaderCache` so that the data loader does not have to load inputs multiple times for each runner.

### Changed
- `Comparator.compare_accuracy` now fails if no outputs were compared.

### Removed
- Removed support for implicit batch ONNX models in `TrtLegacyRunner`. You should use `TrtRunner` for ONNX models instead.


## v0.7.1 (2019-08-29)
### Removed
- Removed `python2` support.

### Fixed
- Bug fixes for TensorFlow Graphs
- Bug fixes for `polygraphy_exec` when using legacy `TrtLegacyRunner`
- Bug fixes for `TrtRunner` for cases with multiple outputs


## v0.7.0 (2019-07-30)
#### Added
- Added support for compression during communication between the runner subprocesses and the main `Comparator` process. This is because `Pipe`s and `Queue`s can only send objects smaller than 2GB.
- Added timeouts to reduce the possibility of hangs in runners.
- Added `--fail-fast` option to `polygraphy_exec` and corresponding `fail_fast` option to `Comparator.compare()`. Useful for determining the first layer at which two models diverge.
- Added `TrtRunner` that can be used to run TRT networks with dynamic shapes. Currently only supports ONNX.

### Changed
- Runners no longer need to specify inputs up front - they can now be specified after `__enter__` is called. This greatly simplifies much of the logic in several runners.
- `RunInfo` no longer contains data about the inputs used.
- `TFOnnxrtRunner` now accepts an opset option when converting graphs to ONNX.


## v0.6.0 (2019-07-17)
- All runner files are now suffixed with `_runner` to disambiguate them from system packages.
- Fixes an issue that prevent EXTRA_VERBOSE logging output from TRT from being displayed.
- Added a `--uff-order` option in case the automatically determined order is wrong.
- Added an experimental `--build-only` option to `polygraphy_exec`
- Comparator will now attempt to permute outputs with mismatched shapes when `check_shapes` is disabled.
- Lowers the default GPU memory fraction, as TensorFlow has OOM issues when it is set too high.
- Added `TFOnnxrtRunner` and `--tfonnx` option to `polygraphy_exec`
- Added `OnnxrtRunner` and moves `TFOnnxrtRunner` into `onnx_runner.py`.
- Added `--save-onnx` option for `OnnxrtRunner`
- Changed `--onnx` `polygraphy_exec` option to `onnxtf` to disambiguate from `--onnxrt`
- Added `CNTKRunner` and `--cntk` option to `polygraphy_exec`
- Changed default shape value to 1. This is the value that is set when no input dimension is specified.
- Added support for loading TF checkpoints.
- Added support for overriding automatically determined outputs in the TF and TF-TRT runners. Added `--tf-outputs` argument to `polygraphy_exec`
- Fixes input shape mismatches between ONNX-RT and TF.
- Added `--plugins` option to `polygraphy_exec` for loading TRT plugins.

## v0.5.0 (2019-04-02)

- Added a function in comparator to perform output validation, and a corresponding flag in `polygraphy_exec`.
- Runners now use OrderedDict for outputs, meaning that the ordering of the outputs will match the order of the layers in the network in most cases.
- Improved TensorFlow output tensor deduction by excluding certain ops that cannot behave like outputs in TensorFlow.
- Version information is now logged at INFO logging severity.
- Removed prepare_inputs/prepare_outputs functions. Instead, runners now do timing on their own in the infer function.
- Changed runner inputs to use dictionaries that map input names to their numpy buffers.
- `polygraphy_exec` will no longer fail if the extension for the model file is unrecognized.
- Added `fp16_mode` option to TfRunner for TF-TRT.

## v0.4.0 (2019-02-28)

- Added an option to limit TensorFlow GPU memory usage
- Added an option to specify minimum segment size to TF-TRT.
- Added an option to write out engine(s) from the TF-TRT graph.
- `polygraphy_exec` now exits when unknown arguments are encountered
- Improves timestamps to be human-readable instead of using seconds from epoch.
- Added support for dynamic ops in TF-TRT
- Added an option to write out tensorboard visualizations.
- Added an option for enabling XLA in the TensorFlow runner.

## v0.3.0 (2019-02-19)

- Added nicer error messages on failed TF-TRT imports
- If a TensorFlow graph specifies a dynamic shape, Polygraphy now automatically populates it with concrete values.
- Added argument groups and moves some unstable arguments to Experimental section.
- Polygraphy will now refuse to write artifacts to the disk if a file already exists wherever it can detect such cases.
- `polygraphy_exec` now emits warnings when unknown command line parameters are used.
- Added capability to write out TensorFlow timelines.
- Changed --save* options to accept directory names instead, and the resulting files are timestamped and named based on the runner name.

## v0.2.0 (2019-02-04)

- Changed command-line parameters to use dashes instead of underscore.
- Modifies TrtLegacyRunner to pass along input order to UFF, instead of permuting the order to CHW.
- Comparator now prints runner output in the same order in which they were specified.
- Added per-inference-inputs command-line arguments for running multiple comparisons.
- Seed is now displayed correctly during Comparator.run().
- User-friendly Comparator output - now suggests command-line flags to get what you were looking for.
- Added layerwise comparison support for TrtLegacyRunner and TfRunner.

## v0.1.2 (2019-01-15)

- Renamed to TRT Polygraphy.

## v0.1.1 (2019-01-15)

- Overhauled README.md
- Modified project structure - created runners, comparator, and logger submodules.
- polygraphy_exec now uses batch size specified by model if none is specified by the user.
- Added framework dependencies to setup.py
- TrtLegacyRunner now displays ONNX parsing errors and exits early on parsing failures.

## v0.1.0 (2019-01-11)

- Initial integration
