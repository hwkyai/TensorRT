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
import copy
import time
from collections import defaultdict

from polygraphy import config, func, mod, util
from polygraphy.logger import G_LOGGER, LogMode

np = mod.lazy_import("numpy")

@mod.export()
class BaseRunner(object):
    """
    Base class for Polygraphy runners. All runners should override the functions and attributes specified here.
    """
    RUNNER_COUNTS = defaultdict(int)

    def __init__(self, name=None, prefix=None):
        """
        Args:
            name (str):
                    The name to use for this runner.
            prefix (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
                    Only used if name is not provided.
        """
        prefix = util.default(prefix, "Runner")
        if name is None:
            count = BaseRunner.RUNNER_COUNTS[prefix]
            BaseRunner.RUNNER_COUNTS[prefix] += 1
            name = "{:}-N{:}-{:}-{:}".format(prefix, count, time.strftime("%x"), time.strftime("%X"))
        self.name = name
        self.inference_time = None

        self.is_active = False
        """bool: Whether this runner has been activated, either via context manager, or by calling ``activate()``."""

        self._cached_input_metadata = None


    @func.constantmethod
    def last_inference_time(self):
        """
        Returns the total inference time required during the last call to ``infer()``.

        Returns:
            float: The time in seconds, or None if runtime was not measured by the runner.
        """
        if self.inference_time is None:
            G_LOGGER.warning("{:35} | inference_time was not set. Inference time will be incorrect!"
                             "To correctly compare runtimes, please set the inference_time property in the"
                             "infer() function".format(self.name), mode=LogMode.ONCE)
            return None
        return self.inference_time


    def __enter__(self):
        """
        Activate the runner for inference. This may involve allocating GPU buffers, for example.
        """
        self.activate()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        """
        Deactivate the runner.

        If the POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS environment variable is set to `1`, this
        will also check that the runner was reset to its state prior to activation.
        """
        self.deactivate()


    def activate_impl(self):
        """
        Implementation for runner activation. Derived classes should override this function
        rather than ``activate()``.
        """
        pass


    def activate(self):
        """
        Activate the runner for inference. This may involve allocating GPU buffers, for example.

        Generally, you should use a context manager instead of manually activating and deactivating.
        For example:
        ::

            with RunnerType(...) as runner:
                runner.infer(...)
        """
        if self.is_active:
            G_LOGGER.warning("{:35} | Already active; will not activate again. If you really want to "
                             "activate this runner again, call activate_impl() directly".format(self.name))
            return

        if config.INTERNAL_CORRECTNESS_CHECKS:
            self._pre_activate_runner_state = copy.copy(vars(self))

        self.activate_impl()
        self.is_active = True


    def infer_impl(self, feed_dict):
        """
        Implementation for runner inference. Derived classes should override this function
        rather than ``infer()``
        """
        raise NotImplementedError("BaseRunner is an abstract class")


    def infer(self, feed_dict, check_inputs=True):
        """
        Runs inference using the provided feed_dict.

        Args:
            feed_dict (OrderedDict[str, numpy.ndarray]):
                    A mapping of input tensor names to corresponding input NumPy arrays.

            check_inputs (bool):
                    Whether to check that the provided ``feed_dict`` includes the expected inputs
                    with the expected data types and shapes.

        Returns:
            OrderedDict[str, numpy.ndarray]:
                    A mapping of output tensor names to their corresponding NumPy arrays.

                    IMPORTANT: Runners may reuse these output buffers. Thus, if you need to save
                    outputs from multiple inferences, you should make a copy with ``copy.deepcopy(outputs)``.
        """
        if not self.is_active:
            G_LOGGER.critical("{:35} | Must be activated prior to calling infer()".format(self.name))

        if check_inputs:
            input_metadata = self.get_input_metadata()
            G_LOGGER.verbose("Runner input metadata is: {:}".format(input_metadata))

            util.check_dict_contains(feed_dict, input_metadata.keys(), dict_name="feed_dict", log_func=G_LOGGER.critical)

            for name, inp in feed_dict.items():
                meta = input_metadata[name]
                if not np.issubdtype(inp.dtype, meta.dtype):
                    G_LOGGER.critical("Input tensor: {:} | Received unexpected dtype: {:}.\n"
                                      "Note: Expected type: {:}".format(name, inp.dtype, meta.dtype))

                if not util.is_valid_shape_override(inp.shape, meta.shape):
                    G_LOGGER.critical("Input tensor: {:} | Received incompatible shape: {:}.\n"
                                      "Note: Expected a shape compatible with: {:}".format(name, inp.shape, meta.shape))

        return self.infer_impl(feed_dict)


    @func.constantmethod
    def get_input_metadata_impl(self):
        """
        Implemenation for `get_input_metadata`. Derived classes should override this function
        rather than `get_input_metadata`.
        """
        raise NotImplementedError("BaseRunner is an abstract class")


    def get_input_metadata(self):
        """
        Returns information about the inputs of the model.
        Shapes here may include dynamic dimensions, represented by ``None``.
        Must be called only after activate() and before deactivate().

        Returns:
            TensorMetadata: Input names, shapes, and data types.
        """
        if self._cached_input_metadata is None:
            self._cached_input_metadata = self.get_input_metadata_impl()
        return self._cached_input_metadata


    def deactivate_impl(self):
        """
        Implementation for runner deactivation. Derived classes should override this function
        rather than ``deactivate()``.
        """
        pass


    def deactivate(self):
        """
        Deactivate the runner.

        If the POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS environment variable is set to `1`, this
        will also check that the runner was reset to its state prior to activation.

        Generally, you should use a context manager instead of manually activating and deactivating.
        For example:
        ::

            with RunnerType(...) as runner:
                runner.infer(...)
        """
        if not self.is_active:
            G_LOGGER.warning("{:35} | Not active; will not deactivate. If you really want to "
                             "deactivate this runner, call deactivate_impl() directly".format(self.name))
            return

        self.inference_time = None
        self._cached_input_metadata = None
        self.is_active = None

        try:
            self.deactivate_impl()
        except:
            raise # Needed so we can have the else clause
        else:
            self.is_active = False
            if config.INTERNAL_CORRECTNESS_CHECKS:
                old_state = self._pre_activate_runner_state
                del self._pre_activate_runner_state
                if old_state != vars(self):
                    G_LOGGER.internal_error("Runner state was not reset after deactivation. "
                                            "Note:\nOld state: {:}\nNew state: {:}".format(old_state, vars(self)))



    def __del__(self):
        if self.is_active:
            # __del__ is not guaranteed to be called, but when it is, this could be a useful warning.
            print("[W] {:35} | Was activated but never deactivated. This could cause a memory leak!".format(self.name))
