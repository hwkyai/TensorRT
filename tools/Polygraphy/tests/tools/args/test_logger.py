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

from polygraphy.logger.logger import G_LOGGER
from polygraphy.tools.args import LoggerArgs
from tests.tools.args.helper import ArgGroupTestHelper
import pytest


VERBOSITY_CASES = {
    "--silent": G_LOGGER.CRITICAL,
    "-qqqqq": G_LOGGER.CRITICAL,
    "-qqqq": G_LOGGER.ERROR,
    "-qqq": G_LOGGER.WARNING,
    "-qq": G_LOGGER.FINISH,
    "-q": G_LOGGER.START,
    "-v": G_LOGGER.VERBOSE,
    "-vv": G_LOGGER.EXTRA_VERBOSE,
    "-vvv": G_LOGGER.SUPER_VERBOSE,
    "-vvvv": G_LOGGER.ULTRA_VERBOSE,
}

class TestLoggerArgs(object):
    @pytest.mark.parametrize("case", VERBOSITY_CASES.items())
    def test_get_logger_verbosities(self, case):
        arg_group = ArgGroupTestHelper(LoggerArgs())
        flag, sev = case

        arg_group.parse_args([flag])
        logger = arg_group.get_logger()

        assert logger.severity == sev


    def test_logger_log_file(self):
        arg_group = ArgGroupTestHelper(LoggerArgs())

        arg_group.parse_args(["--log-file=fake_log_file.log"])
        logger = arg_group.get_logger()
        assert logger.log_file == "fake_log_file.log"
