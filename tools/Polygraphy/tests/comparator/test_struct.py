

import numpy as np
import pytest
import contextlib
from polygraphy import config
from polygraphy.comparator import IterationResult, RunResults
from polygraphy.comparator.struct import LazyNumpyArray
from polygraphy.exception import PolygraphyException


def make_iter_results(runner_name):
    return [IterationResult(outputs={"dummy_out": np.zeros((4, 4))}, runner_name=runner_name)] * 2


@pytest.fixture(scope="session")
def run_results():
    results = RunResults()
    results.append((
        "runner0",
        make_iter_results("runner0")
    ))
    results.append((
        "runner1",
        make_iter_results("runner1")
    ))
    return results


class TestRunResults(object):
    def test_items(self, run_results):
        for name, iteration_results in run_results.items():
            assert isinstance(name, str)
            assert isinstance(iteration_results, list)
            for iter_res in iteration_results:
                assert isinstance(iter_res, IterationResult)


    def test_keys(self, run_results):
        assert list(run_results.keys()) == ["runner0", "runner1"]


    def test_values(self, run_results):
        for iteration_results in run_results.values():
            for iter_res in iteration_results:
                assert isinstance(iter_res, IterationResult)


    def test_getitem(self, run_results):
        assert isinstance(run_results["runner0"][0], IterationResult)
        assert isinstance(run_results[0][1][0], IterationResult)
        assert run_results[0][1] == run_results["runner0"]
        assert run_results[1][1] == run_results["runner1"]


    def test_getitem_out_of_bounds(self, run_results):
        with pytest.raises(IndexError):
            run_results[2]

        with pytest.raises(PolygraphyException, match="does not exist in this"):
            run_results["runner2"]


    def test_setitem(self, run_results):
        def check_results(results, is_none=False):
            for iter_res in results["runner1"]:
                if is_none:
                    assert not iter_res
                    assert iter_res.runner_name == ""
                else:
                    assert iter_res
                    assert iter_res.runner_name

        check_results(run_results)

        iter_results = [IterationResult(outputs=None, runner_name=None)]
        run_results["runner1"] = iter_results

        check_results(run_results, is_none=True)


    def test_setitem_out_of_bounds(self, run_results):
        iter_results = [IterationResult(outputs=None, runner_name="new")]
        run_results["runner2"] = iter_results

        assert len(run_results) == 3
        assert run_results["runner2"][0].runner_name == "new"


    def test_contains(self, run_results):
        assert "runner0" in run_results
        assert "runner1" in run_results
        assert "runner3" not in run_results


class TestLazyNumpyArray(object):
    @pytest.mark.parametrize("set_threshold", [True, False])
    def test_unswapped_array(self, set_threshold):
        with contextlib.ExitStack() as stack:
            if set_threshold:
                def reset_array_swap():
                    config.ARRAY_SWAP_THRESHOLD_MB = -1
                stack.callback(reset_array_swap)

                config.ARRAY_SWAP_THRESHOLD_MB = 8

            small_shape = (7 * 1024 * 1024, )
            small_array = np.ones(shape=small_shape, dtype=np.byte)
            lazy = LazyNumpyArray(small_array)
            assert np.array_equal(small_array, lazy.arr)
            assert lazy.tmpfile is None

            assert np.array_equal(small_array, lazy.numpy())


    def test_swapped_array(self):
        with contextlib.ExitStack() as stack:
            def reset_array_swap():
                config.ARRAY_SWAP_THRESHOLD_MB = -1
            stack.callback(reset_array_swap)

            config.ARRAY_SWAP_THRESHOLD_MB = 8

            large_shape = (9 * 1024 * 1024, )
            large_array = np.ones(shape=large_shape, dtype=np.byte)
            lazy = LazyNumpyArray(large_array)
            assert lazy.arr is None
            assert lazy.tmpfile is not None

            assert np.array_equal(large_array, lazy.numpy())
