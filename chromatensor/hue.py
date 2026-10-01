"""Native RGBD hue tensors. Operators call Rust and never reconstruct real scalars."""
from __future__ import annotations

import ctypes as C
from contextlib import ExitStack
import hashlib
from importlib.util import find_spec
import math
import operator
import os
from pathlib import Path
import platform
import threading
from functools import lru_cache, wraps

PHASE_MODULUS = 391680


def _platform_directory():
    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64"}:
        raise RuntimeError("Native hue binaries currently support x86-64 Windows and Linux")
    if os.name == "nt":
        return "windows-x86_64", "hue_tensor.dll"
    if platform.system() == "Linux":
        return "linux-x86_64", "libhue_tensor.so"
    raise RuntimeError("Build the Rust hue library for this operating system before using HueTensor")


def _library_path():
    directory, filename = _platform_directory()
    path = Path(__file__).resolve().parent / "_native" / directory / filename
    if path.is_file():
        return path
    # The portable research wheel contains Python code only. A separately
    # installed platform wheel can supply its native library.
    standalone = find_spec("chromatic_hue")
    if standalone is not None and standalone.origin:
        path = Path(standalone.origin).resolve().parent / "_native" / directory / filename
        if path.is_file():
            return path
    raise RuntimeError("Native hue library is missing. Run python scripts/build_native_hue.py in a checkout or install the chromatic-hue platform wheel.")


@lru_cache(maxsize=1)
def _library():
    path = _library_path()
    lib = C.CDLL(str(path))
    ptr, size, u32, u8 = C.c_void_p, C.c_size_t, C.c_uint32, C.c_uint8
    signatures = {
        "ht_abi_version": ([], u32), "ht_modulus": ([], u32),
        "ht_last_error": ([], C.c_char_p), "ht_free": ([ptr], None),
        "ht_len": ([ptr], size), "ht_ndim": ([ptr], size),
        "ht_new_phases": ([C.POINTER(u32), size, C.POINTER(size), size, C.POINTER(ptr)], C.c_int32),
        "ht_new_rgbd": ([C.POINTER(u8), size, C.POINTER(size), size, C.POINTER(ptr)], C.c_int32),
        "ht_new_rgb": ([C.POINTER(u8), size, C.POINTER(size), size, C.POINTER(ptr)], C.c_int32),
        "ht_copy_shape": ([ptr, C.POINTER(size), size], C.c_int32),
        "ht_copy_tokens": ([ptr, C.POINTER(u8), size], C.c_int32),
        "ht_copy_rgb": ([ptr, C.POINTER(u8), size], C.c_int32),
        "ht_copy_phases": ([ptr, C.POINTER(u32), size], C.c_int32),
        "ht_binary": ([ptr, ptr, u32, C.POINTER(ptr)], C.c_int32),
        "ht_rotate": ([ptr, C.c_int64, C.POINTER(ptr)], C.c_int32),
        "ht_blend": ([ptr, ptr, u8, C.POINTER(ptr)], C.c_int32),
        "ht_matmul": ([ptr, ptr, C.POINTER(ptr)], C.c_int32),
        "ht_dot": ([ptr, ptr, C.POINTER(ptr)], C.c_int32),
        "ht_sum": ([ptr, size, C.POINTER(ptr)], C.c_int32),
        "ht_reshape": ([ptr, C.POINTER(size), size, C.POINTER(ptr)], C.c_int32),
        "ht_permute": ([ptr, C.POINTER(size), size, C.POINTER(ptr)], C.c_int32),
        "ht_slice": ([ptr, size, size, size, size, C.POINTER(ptr)], C.c_int32),
        "ht_save": ([ptr, C.c_char_p], C.c_int32),
        "ht_load": ([C.c_char_p, size, C.POINTER(ptr)], C.c_int32),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    if lib.ht_abi_version() != 1 or lib.ht_modulus() != PHASE_MODULUS:
        raise RuntimeError("Native hue library ABI or hue wheel differs from the binding")
    return lib, path


def native_info():
    lib, path = _library()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size, "abi_version": lib.ht_abi_version(),
            "phase_modulus": lib.ht_modulus(), "runtime": "Rust integer color-token operators"}


def _check(status):
    if status:
        message = _library()[0].ht_last_error()
        text = message.decode("utf-8", "replace") if message else "Native hue operation failed"
        if status == 2:
            raise OSError(text)
        if status == 3:
            raise RuntimeError(text)
        raise ValueError(text)


def _index(value, name="integer"):
    if isinstance(value, bool):
        raise TypeError(name + " must be an integer, not a Boolean")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(name + " must be an integer") from error


def _shape(shape):
    result = tuple(_index(value, "shape dimension") for value in shape)
    if any(value < 0 or value > C.c_size_t(-1).value for value in result):
        raise ValueError("Shape dimensions must fit nonnegative size_t values")
    if math.prod(result) > C.c_size_t(-1).value:
        raise ValueError("Shape product overflows size_t")
    return result


def _flatten(values):
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        if not values:
            return [], (0,)
        children = [_flatten(value) for value in values]
        if any(shape != children[0][1] for _, shape in children):
            raise ValueError("Phase values must be rectangular")
        return [item for flat, _ in children for item in flat], (len(values),) + children[0][1]
    if isinstance(values, (bytes, bytearray, memoryview)):
        return list(values), (len(values),)
    try:
        value = _index(values, "phase")
    except TypeError:
        if hasattr(values, "__iter__"):
            return _flatten(list(values))
        raise
    return [value], ()


def _owned(method):
    """Keep each native handle alive while ctypes releases the GIL."""
    @wraps(method)
    def call(self, *args, **kwargs):
        objects = {id(value): value for value in (self, *args, *kwargs.values())
                   if isinstance(value, HueTensor)}
        with ExitStack() as stack:
            for _, value in sorted(objects.items()):
                lock = getattr(value, "_lock", None)
                if lock is not None:
                    stack.enter_context(lock)
            return method(self, *args, **kwargs)
    return call


class HueTensor:
    """An owned Rust tensor of canonical RGB hues and D sub-hue bytes.

    + composes phases, * multiplies phases in Z/391680Z, and @ contracts
    that algebra. These operators do not implement real-valued arithmetic.
    """

    execution = {"storage": "native RGBD bytes", "operators": "Rust integer hue algebra",
                 "floating_point_decode": False, "autograd": False}

    def __init__(self, *args, **kwargs):
        raise TypeError("Use HueTensor.from_phases, from_rgbd, from_rgb or load")

    @classmethod
    def _take(cls, pointer):
        if not pointer.value:
            raise RuntimeError("Native operation returned a null tensor")
        instance = object.__new__(cls)
        instance._lock = threading.RLock()
        instance._handle = pointer
        return instance

    @classmethod
    def _call(cls, name, *args):
        output = C.c_void_p()
        _check(getattr(_library()[0], name)(*args, C.byref(output)))
        return cls._take(output)

    @classmethod
    def from_phases(cls, phases, shape=None):
        source_shape = getattr(phases, "shape", None)
        source_dtype = getattr(phases, "dtype", None)
        if hasattr(source_dtype, "kind") and source_dtype.kind not in {"i", "u"}:
            raise TypeError("Phase arrays must have an integer dtype")
        values, inferred = _flatten(phases)
        dimensions = _shape((source_shape if source_shape is not None else inferred) if shape is None else shape)
        if math.prod(dimensions) != len(values):
            raise ValueError("Shape does not match the phase count")
        if any(value < 0 or value >= PHASE_MODULUS for value in values):
            raise ValueError("Phase must be between zero and 391679")
        data = (C.c_uint32 * len(values))(*values)
        dims = (C.c_size_t * len(dimensions))(*dimensions)
        return cls._call("ht_new_phases", data, len(values), dims, len(dimensions))

    @classmethod
    def _from_bytes(cls, name, data, shape, channels):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Color input must be a byte buffer")
        data = bytes(data)
        dimensions = _shape(shape)
        if len(data) != channels * math.prod(dimensions):
            raise ValueError("Shape does not match the color byte count")
        buffer = (C.c_uint8 * len(data)).from_buffer_copy(data)
        dims = (C.c_size_t * len(dimensions))(*dimensions)
        return cls._call(name, buffer, len(data), dims, len(dimensions))

    @classmethod
    def from_rgbd(cls, data, shape):
        return cls._from_bytes("ht_new_rgbd", data, shape, 4)

    @classmethod
    def from_rgb(cls, data, shape):
        """Project RGB hues onto the saturated wheel. Gray has no hue."""
        return cls._from_bytes("ht_new_rgb", data, shape, 3)

    def _ptr(self):
        pointer = getattr(self, "_handle", None)
        if pointer is None or not pointer.value:
            raise RuntimeError("HueTensor has been closed")
        return pointer

    @_owned
    def close(self):
        pointer = getattr(self, "_handle", None)
        if pointer is not None and pointer.value:
            _library()[0].ht_free(pointer)
            self._handle = C.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @_owned
    def __enter__(self):
        self._ptr()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    @_owned
    def shape(self):
        lib = _library()[0]
        ndim = lib.ht_ndim(self._ptr())
        output = (C.c_size_t * ndim)()
        _check(lib.ht_copy_shape(self._ptr(), output, ndim))
        return tuple(output)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    @_owned
    def numel(self):
        return _library()[0].ht_len(self._ptr())

    @property
    def storage_nbytes(self):
        return self.numel * 4

    @property
    @_owned
    def rgbd_bytes(self):
        output = (C.c_uint8 * self.storage_nbytes)()
        _check(_library()[0].ht_copy_tokens(self._ptr(), output, len(output)))
        return bytes(output)

    @property
    @_owned
    def rgb_bytes(self):
        output = (C.c_uint8 * (self.numel * 3))()
        _check(_library()[0].ht_copy_rgb(self._ptr(), output, len(output)))
        return bytes(output)

    @_owned
    def phases(self):
        output = (C.c_uint32 * self.numel)()
        _check(_library()[0].ht_copy_phases(self._ptr(), output, len(output)))
        return tuple(output)

    @_owned
    def _binary(self, other, opcode):
        if not isinstance(other, HueTensor):
            return NotImplemented
        return self._call("ht_binary", self._ptr(), other._ptr(), opcode)

    def __add__(self, other):
        return self._binary(other, 0)

    def __sub__(self, other):
        return self._binary(other, 1)

    def __mul__(self, other):
        return self._binary(other, 2)

    @_owned
    def __matmul__(self, other):
        if not isinstance(other, HueTensor):
            return NotImplemented
        return self._call("ht_matmul", self._ptr(), other._ptr())

    @_owned
    def rotate(self, ticks):
        ticks = _index(ticks, "rotation")
        if not -(2**63) <= ticks < 2**63:
            raise ValueError("Rotation must fit signed 64 bits")
        return self._call("ht_rotate", self._ptr(), ticks)

    @_owned
    def blend(self, other, weight=128):
        if not isinstance(other, HueTensor):
            raise TypeError("Blend requires another HueTensor")
        weight = _index(weight, "blend weight")
        if not 0 <= weight <= 255:
            raise ValueError("Blend weight must be between zero and 255")
        return self._call("ht_blend", self._ptr(), other._ptr(), weight)

    @_owned
    def dot(self, other):
        if not isinstance(other, HueTensor):
            raise TypeError("Dot requires another HueTensor")
        return self._call("ht_dot", self._ptr(), other._ptr())

    @_owned
    def sum(self, axis=None):
        if axis is None:
            return self.reshape(self.numel).sum(0)
        axis = _index(axis, "axis")
        if not -self.ndim <= axis < self.ndim:
            raise ValueError("Reduction axis is outside the tensor")
        return self._call("ht_sum", self._ptr(), axis % self.ndim)

    @_owned
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        dimensions = _shape(shape)
        dims = (C.c_size_t * len(dimensions))(*dimensions)
        return self._call("ht_reshape", self._ptr(), dims, len(dimensions))

    @_owned
    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = tuple(axes[0])
        axes = tuple(_index(axis, "axis") for axis in axes)
        ndim = self.ndim
        if len(axes) != ndim or any(not -ndim <= axis < ndim for axis in axes):
            raise ValueError("Permutation must contain each axis exactly once")
        axes = tuple(axis % ndim for axis in axes)
        if len(set(axes)) != ndim:
            raise ValueError("Permutation must contain each axis exactly once")
        dims = (C.c_size_t * ndim)(*axes)
        return self._call("ht_permute", self._ptr(), dims, ndim)

    @property
    def T(self):
        return self.permute(*reversed(range(self.ndim)))

    @_owned
    def slice(self, axis, start=0, stop=None, step=1):
        axis = _index(axis, "axis")
        if not -self.ndim <= axis < self.ndim:
            raise ValueError("Slice axis is outside the tensor")
        axis %= self.ndim
        start, step = _index(start, "start"), _index(step, "step")
        stop = self.shape[axis] if stop is None else _index(stop, "stop")
        if not 0 < step <= C.c_size_t(-1).value or not 0 <= start <= stop <= self.shape[axis]:
            raise ValueError("Slice requires a positive step and bounds inside its axis")
        return self._call("ht_slice", self._ptr(), axis, start, stop, step)

    @_owned
    def save(self, path):
        path = Path(path)
        encoded = os.fsencode(path)
        if b"\x00" in encoded:
            raise ValueError("Hue file paths cannot contain NUL bytes")
        _check(_library()[0].ht_save(self._ptr(), encoded))
        return path.stat().st_size

    @classmethod
    def load(cls, path, max_bytes=256 * 1024**2):
        budget = _index(max_bytes, "file budget")
        if not 0 <= budget <= C.c_size_t(-1).value:
            raise ValueError("File budget must fit nonnegative size_t")
        encoded = os.fsencode(Path(path))
        if b"\x00" in encoded:
            raise ValueError("Hue file paths cannot contain NUL bytes")
        return cls._call("ht_load", encoded, budget)

    def __repr__(self):
        return f"HueTensor(shape={self.shape}, payload_bytes={self.storage_nbytes}, runtime='Rust')"


ColorTensor = HueTensor
Tensor = HueTensor


class HueEncoder:
    """Boundary encoder implemented by the native Rust color library."""

    encode_rgb = staticmethod(HueTensor.from_rgb)
    encode_rgbd = staticmethod(HueTensor.from_rgbd)
    encode_phases = staticmethod(HueTensor.from_phases)


class HueDecoder:
    """Decode native hue tokens to color bytes or finite integer phase ticks."""

    @staticmethod
    def decode_rgb(tensor):
        if not isinstance(tensor, HueTensor):
            raise TypeError("HueDecoder requires a HueTensor")
        return tensor.rgb_bytes

    @staticmethod
    def decode_phases(tensor):
        if not isinstance(tensor, HueTensor):
            raise TypeError("HueDecoder requires a HueTensor")
        return tensor.phases()


__all__ = ["HueTensor", "ColorTensor", "Tensor", "HueEncoder", "HueDecoder", "PHASE_MODULUS", "native_info"]
