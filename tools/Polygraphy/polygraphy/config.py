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
import os


INTERNAL_CORRECTNESS_CHECKS = bool(os.environ.get("POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS", "0") != "0")
"""
Whether internal correctness checks are enabled.
This can be configured by setting the 'POLYGRAPHY_INTERNAL_CORRECTNESS_CHECKS' environment variable.
"""

AUTOINSTALL_DEPS = bool(os.environ.get("POLYGRAPHY_AUTOINSTALL_DEPS", "0") != "0")
"""
Whether Polygraphy will automatically install required Python packages at runtime.
This can be configured by setting the 'POLYGRAPHY_AUTOINSTALL_DEPS' environment variable.
"""

ARRAY_SWAP_THRESHOLD_MB = int(os.environ.get("POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB", "-1"))
"""
The threshold, in megabytes, above which Polygraphy will evict a NumPy array from memory and swap it to disk.
A negative value disables swapping and a value of 0 causes all arrays to be saved to disk.
Disabled by default.
This can be configured by setting the 'POLYGRAPHY_ARRAY_SWAP_THRESHOLD_MB' environment variable.
"""
