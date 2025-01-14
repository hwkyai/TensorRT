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

import functools
import io
import json
from collections import OrderedDict

from polygraphy import config, constants, mod
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")
util = mod.lazy_import("polygraphy.util.util")

TYPE_STRING_PREFIX = "__polygraphy_encoded_"

def str_from_type(typ):
    return TYPE_STRING_PREFIX + typ.__name__


class BaseCustomImpl(object):
    """
    Base class for Polygraphy's JSON encoder/decoder.
    """
    @classmethod
    def register(cls, typ):
        """
        Decorator that registers JSON encoding/decoding functions for types.

        For the documentation that follows, assume we have a class:
        ::

            class Dummy(object):
                def __init__(self, x):
                    self.x = x

        ========
        Encoders
        ========

        Encoder functions should accept instances of the specified type and
        return dictionaries.

        For example:
        ::

            @Encoder.register(Dummy)
            def encode(dummy):
                return {"x": dummy.x}


        To use the custom encoder, use the `to_json` helper:
        ::

            d = Dummy(x=1)
            d_json = to_json(d)


        ========
        Decoders
        ========

        Decoder functions should accept dictionaries, and return instances of the
        type.

        For example:
        ::

            @Decoder.register(Dummy)
            def decode(dct):
                return Dummy(x=dct["x"])


        To use the custom decoder, use the `from_json` helper:
        ::

            from_json(d_json)


        Args:
            typ (type): The type of the class for which to register the function.
        """
        def register_impl(func):
            def add(key, val):
                if key in cls.polygraphy_registered:
                    G_LOGGER.critical("Duplicate serialization function for type: {:}.\n"
                                      "Note: Existing function: {:}, New function: {:}".format(
                                        key, cls.polygraphy_registered[key], func))
                cls.polygraphy_registered[key] = val


            if cls == Encoder:
                def wrapped(obj):
                    dct = func(obj)
                    dct[str_from_type(typ)] = constants.TYPE_MARKER
                    return dct

                add(typ, wrapped)
                return wrapped
            elif cls == Decoder:
                def wrapped(dct):
                    del dct[str_from_type(typ)]
                    return func(dct)

                add(str_from_type(typ), wrapped)
            else:
                G_LOGGER.critical("Cannot register for unrecognized class type: ")


        return register_impl


@mod.export()
class Encoder(BaseCustomImpl, json.JSONEncoder):
    """
    Polygraphy's custom JSON Encoder implementation.
    """
    polygraphy_registered = {}

    def default(self, o):
        if type(o) in self.polygraphy_registered:
            return self.polygraphy_registered[type(o)](o)
        return super().default(o)


@mod.export()
class Decoder(BaseCustomImpl):
    """
    Polygraphy's custom JSON Decoder implementation.
    """
    polygraphy_registered = {}

    def __call__(self, pairs):
        dct = OrderedDict(pairs)

        if config.INTERNAL_CORRECTNESS_CHECKS:
            custom_type_keys = [key for key in dct if key.startswith(TYPE_STRING_PREFIX)]
            if custom_type_keys and custom_type_keys[0] not in self.polygraphy_registered:
                G_LOGGER.internal_error("Custom type has no decode function registered! "
                                        "Note: Encoded object is:\n{:}".format(dct))

        # The encoder will insert special key-value pairs into dictionaries encoded from
        # custom types. If we find one, then we know to decode using the corresponding custom
        # type function.
        for type_str, func in self.polygraphy_registered.items():
            if type_str in dct and dct[type_str] == constants.TYPE_MARKER: # Found a custom type!
                return func(dct)
        return dct


NUMPY_REGISTRATION_SUCCESS = False
def try_register_numpy_json(func):
    """
    Decorator that attempts to register JSON encode/decode methods
    for numpy arrays if NumPy is available and the methods have not already been registered.

    This needs to be attempted multiple times because numpy may become available in the
    middle of execution - for example, if using dependency auto-installation.
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        global NUMPY_REGISTRATION_SUCCESS
        if not NUMPY_REGISTRATION_SUCCESS and mod.has_mod(np, "__version__"):
            # We define this along-side load_json/save_json so that it is guaranteed to be
            # imported before we need to encode/decode NumPy arrays.
            @Encoder.register(np.ndarray)
            def encode(array):
                outfile = io.BytesIO()
                np.savez(outfile, array)
                outfile.seek(0)
                return {
                    "array": outfile.read().decode('latin-1')
                }


            @Decoder.register(np.ndarray)
            def decode(dct):
                infile = io.BytesIO(dct["array"].encode('latin-1'))
                # We always encode arrays separately.
                return list(np.load(infile, allow_pickle=False).values())[0]


            NUMPY_REGISTRATION_SUCCESS = True
        return func(*args, **kwargs)
    return wrapped


@mod.export()
@try_register_numpy_json
def to_json(obj):
    """
    Encode an object to JSON.

    NOTE: For Polygraphy objects, you should use the ``to_json()`` method instead.

    Returns:
        str: A JSON representation of the object.
    """
    return json.dumps(obj, cls=Encoder, indent=constants.TAB)


@mod.export()
@try_register_numpy_json
def from_json(src):
    """
    Decode a JSON string to an object.

    NOTE: For Polygraphy objects, you should use the ``from_json()`` method instead.

    Args:
        src (str):
                The JSON representation of the object

    Returns:
        object: The decoded instance
    """
    return json.loads(src, object_pairs_hook=Decoder())


@mod.export_deprecated_alias("pickle_save", remove_in="0.31.0", use_instead="JSON serialization. "
                             "This function has been migrated to use JSON and will NOT pickle the input object. "
                             "Use save_json")
@mod.export()
@try_register_numpy_json
def save_json(obj, dest, description=None):
    """
    Encode an object as JSON and save it to a file.

    NOTE: For Polygraphy objects, you should use the ``save()`` method instead.

    Args:
        obj (object): The object to save.
        src (Union[str, file-like]): The path or file-like object to save to.
    """
    util.save_file(to_json(obj), dest, mode="w", description=description)


@mod.export_deprecated_alias("pickle_load", remove_in="0.31.0", use_instead="load_json")
@mod.export()
@try_register_numpy_json
def load_json(src, description=None):
    """
    Loads a file and decodes the JSON contents.

    NOTE: For Polygraphy objects, you should use the ``load()`` method instead.

    Args:
        src (Union[str, file-like]): The path or file-like object to load from.

    Returns:
        object: The object, or `None` if nothing could be read.
    """
    try:
        return from_json(util.load_file(src, mode="r", description=description))
    except UnicodeDecodeError:
        # This is a pickle file from Polygraphy 0.26.1 or older.
        mod.warn_deprecated("pickle", use_instead="JSON", remove_in="0.31.0")
        G_LOGGER.critical("It looks like you're trying to load a Pickle file.\nPolygraphy migrated to using JSON "
                          "instead of Pickle in version 0.27.0 for security reasons.\nYou can convert your existing "
                          "pickled data to JSON using the command-line tool: `polygraphy to-json {:} -o new.json`.\nAll data serialized "
                          "from this and future versions of Polygraphy will always use JSON. ".format(src))


@mod.export()
def add_json_methods(description=None):
    """
    Decorator that adds 4 JSON helper methods to a class:

    - to_json(): Convert to JSON string
    - from_json(): Convert from JSON string
    - save(): Convert to JSON and save to file
    - load(): Load from file and convert from JSON

    Args:
        description (str):
                A description of what is being saved or loaded.
    """
    def add_json_methods_impl(cls):
        # JSON methods

        def check_decoded(obj):
            if not isinstance(obj, cls):
                G_LOGGER.critical("Provided JSON cannot be decoded into a {:}.\n"
                                  "Note: JSON was decoded into a {:}:\n{:}".format(cls.__name__, type(obj), obj))
            return obj


        def _to_json_method(self):
            """
            Encode this instance as a JSON object.

            Returns:
                str: A JSON representation of this instance.
            """
            return to_json(self)


        def _from_json_method(src):
            return check_decoded(from_json(src))


        _from_json_method.__doc__ = """
            Decode a JSON object and create an instance of this class.

            Args:
                src (str):
                        The JSON representation of the object

            Returns:
                {cls}: The decoded instance

            Raises:
                PolygraphyException:
                        If the JSON cannot be decoded to an instance of {cls}
            """.format(cls=cls.__name__)


        cls.to_json = _to_json_method
        cls.from_json = staticmethod(_from_json_method)

        # Save/Load methods

        def _save_method(self, dest):
            """
            Encode this instance as a JSON object and save it to the specified path
            or file-like object.

            Args:
                dest (Union[str, file-like]):
                      The path or file-like object to write to.

            """
            save_json(self, dest, description=description)


        def _load_method(src):
            return check_decoded(load_json(src, description=description))


        _load_method.__doc__ = """
            Loads an instance of this class from a JSON file.

            Args:
                src (Union[str, file-like]): The path or file-like object to read from.

            Returns:
                {cls}: The decoded instance

            Raises:
                PolygraphyException:
                        If the JSON cannot be decoded to an instance of {cls}
            """.format(cls=cls.__name__)


        cls.save = _save_method
        cls.load = staticmethod(_load_method)

        return cls

    return add_json_methods_impl
