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

from polygraphy import constants, mod, util
from polygraphy.common import TensorMetadata
from polygraphy.logger import G_LOGGER
from polygraphy.tools.script import Script, ensure_safe, inline, safe

np = mod.lazy_import("numpy")


@mod.export()
def cast(val):
    """
    Cast a value from a string to one of:
    [int, float, str, List[int], List[float], List[str]]

    Args:
        val (str): The value to cast.

    Returns:
        object: The casted value.
    """
    val = str(val.strip())

    if val.strip("[]") != val:
        return [cast(elem) for elem in val.strip("[]").split(",")]

    try:
        return int(val) # This fails for float strings like '0.0'
    except:
        pass

    try:
        return float(val) # This fails for non-numerical strings like 'isildur'
    except:
        pass
    return val.strip("\"'")


@mod.export()
def run_script(script_func, *args):
    """
    Populates a script using the provided callable, then returns
    the variable indicated by the return value of the callable.

    Args:
        script_func (Callable(Script, *args) -> str):
                A callable that populates a Script and then returns
                the name of an object defined within the script to retrieve.
        args:
                Additional positional argruments to pass to script_func.
                The script_func should accept these by variable name instead
                of taking the values themselves. Values of ``None`` will be
                passed directly instead of by variable name.

    Returns:
        object:
                An object defined within the script, or ``None`` if it is not
                defined by the script.
    """
    script = Script()

    arg_names = []
    for index, arg in enumerate(args):
        if arg is not None:
            arg_name = safe("__arg{:}", index)
            locals()[arg_name.unwrap()] = arg
            arg_names.append(inline(arg_name))
        else:
            arg_names.append(None)

    safe_ret_name = script_func(script, *arg_names)
    exec(str(script), globals(), locals())

    if safe_ret_name is not None:
        ret_name = ensure_safe(safe_ret_name).unwrap()
        if ret_name in locals():
            return locals()[ret_name]
    return None


@mod.export()
def get(args, attr):
    """
    Gets a command-line argument if it exists, otherwise returns None.

    Args:
        args: The command-line arguments.
        attr (str): The name of the command-line argument.
    """
    if hasattr(args, attr):
        return getattr(args, attr)
    return None


@mod.export()
def get_outputs(args, name):
    outputs = get(args, name)
    if outputs is not None and len(outputs) == 2 and outputs == ["mark", "all"]:
        outputs = constants.MARK_ALL
    return outputs


@mod.export()
def get_outputs_for_script(script, outputs):
    if outputs == constants.MARK_ALL:
        script.add_import(["constants"], frm="polygraphy")
        outputs = inline(safe("constants.MARK_ALL"))
    return outputs


def np_types():
    """
    Returns a list of human-readable names of NumPy data types.
    """
    return sorted(set(np.dtype(dtype).name for dtype in np.sctypeDict.values()))


def np_type_from_str(dt_str):
    """
    Converts a string representation of a data type to a NumPy data type.

    Args:
        dt_str (str): The string representation of the data type.

    Returns:
        np.dtype: The NumPy data type.

    Raises:
        KeyError: If the provided string does not correspond to a NumPy data type.
    """
    try:
        return {np.dtype(dtype).name: np.dtype(dtype) for dtype in np.sctypeDict.values()}[dt_str]
    except KeyError:
        G_LOGGER.error("Could not understand data type: {:}. Did you forget to specify a data type? "
                         "Please use one of: {:} or `auto`.".format(dt_str, np_types()))
        raise


@mod.export()
def parse_dict_with_default(arg_lst, cast_to=None, sep=None):
    """
    Generate a dictionary from a list of arguments of the form:
    ``<key>:<val>``. If ``<key>`` is empty, the value will be assigned
    to an empty string key in the returned mapping.

    Args:
        arg_lst (List[str]):
                The arguments to map.

        cast_to (type):
                The type to cast the values in the map. By default,
                uses the type returned by ``cast``.
        sep (str):
                The separator between the key and value strings.
    Returns:
        Dict[str, obj]: The mapping.
    """
    sep = util.default(sep, ":")


    if arg_lst is None:
        return

    arg_map = {}
    for arg in arg_lst:
        key, _, val = arg.rpartition(sep)
        val = cast(val)
        if cast_to:
            val = cast_to(val)
        arg_map[key] = val
    return arg_map


@mod.deprecate(remove_in="0.35.0", use_instead=": as a separator and write shapes in the form [dim0,...,dimN]", name="Using , as a separator")
def parse_meta_legacy(meta_args, includes_shape=True, includes_dtype=True):
    """
    Parses a list of tensor metadata arguments of the form "<name>,<shape>,<dtype>"
    `shape` and `dtype` are optional, but `dtype` must always come after `shape` if they are both enabled.

    Args:
        meta_args (List[str]): A list of tensor metadata arguments from the command-line.
        includes_shape (bool): Whether the arguments include shape information.
        includes_dtype (bool): Whether the arguments include dtype information.

    Returns:
        TensorMetadata: The parsed tensor metadata.
    """
    SEP = ","
    SHAPE_SEP = "x"
    meta = TensorMetadata()
    for orig_tensor_meta_arg in meta_args:
        tensor_meta_arg = orig_tensor_meta_arg

        def pop_meta(name):
            nonlocal tensor_meta_arg
            tensor_meta_arg, _, val = tensor_meta_arg.rpartition(SEP)
            if not tensor_meta_arg:
                G_LOGGER.exit("Could not parse {:} from argument: {:}. Is it separated by a comma "
                                    "(,) from the tensor name?".format(name, orig_tensor_meta_arg))
            if val.lower() == "auto":
                val = None
            return val


        def parse_dtype(dtype):
            if dtype is not None:
                dtype = np_type_from_str(dtype)
            return dtype


        def parse_shape(shape):
            if shape is not None:
                def parse_shape_dim(buf):
                    try:
                        buf = int(buf)
                    except:
                        pass
                    return buf


                parsed_shape = []
                # Allow for quoted strings in shape dimensions
                in_quotes = False
                buf = ""
                for char in shape.lower():
                    if char in ["\"", "'"]:
                        in_quotes = not in_quotes
                    elif not in_quotes and char == SHAPE_SEP:
                        parsed_shape.append(parse_shape_dim(buf))
                        buf = ""
                    else:
                        buf += char
                # For the last dimension
                if buf:
                    parsed_shape.append(parse_shape_dim(buf))
                shape = tuple(parsed_shape)
            return shape


        name = None
        dtype = None
        shape = None

        if includes_dtype:
            dtype = parse_dtype(pop_meta("data type"))

        if includes_shape:
            shape = parse_shape(pop_meta("shape"))

        name = tensor_meta_arg
        meta.add(name, dtype, shape)
    return meta


def parse_meta_new_impl(meta_args, includes_shape=True, includes_dtype=True):
    SEP = ":"
    meta = TensorMetadata()
    for meta_arg in meta_args:
        name, shape, dtype = None, None, None

        def pop_meta(func):
            nonlocal meta_arg
            meta_arg, _, val = meta_arg.rpartition(SEP)
            val = cast(val.strip())
            if isinstance(val, str) and val.lower() == "auto":
                return None
            return func(val)

        if includes_dtype:
            dtype = pop_meta(func=np_type_from_str)

        if includes_shape:
            shape = pop_meta(func=lambda s: tuple(e for e in s if e != ""))

        name = meta_arg

        meta.add(name, dtype=dtype, shape=shape)
    return meta


@mod.export()
def parse_meta(meta_args, includes_shape=True, includes_dtype=True):
    """
    Parses a list of tensor metadata arguments of the form "<name>:<shape>:<dtype>"
    `shape` and `dtype` are optional, but `dtype` must always come after `shape` if they are both enabled.

    Args:
        meta_args (List[str]): A list of tensor metadata arguments from the command-line.
        includes_shape (bool): Whether the arguments include shape information.
        includes_dtype (bool): Whether the arguments include dtype information.

    Returns:
        TensorMetadata: The parsed tensor metadata.
    """
    if all((includes_shape and "[" in arg) or (includes_dtype and "," not in arg) for arg in meta_args):
        return parse_meta_new_impl(meta_args, includes_shape, includes_dtype)
    return parse_meta_legacy(meta_args, includes_shape, includes_dtype)
