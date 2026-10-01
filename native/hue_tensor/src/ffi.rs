//! Portable C calls. Callers must supply readable input buffers and writable
//! output buffers of the stated size, and valid handles returned here.

use crate::{Error, Result, Tensor, Token, ABI_VERSION, MAX_ELEMENTS, MAX_RANK, MODULUS};
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;

thread_local! { static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }

fn set_error(message: &str) {
    let clean = message.replace('\0', "?");
    LAST_ERROR.with(|slot| *slot.borrow_mut() = CString::new(clean).unwrap());
}

fn boundary<F: FnOnce() -> Result<()>>(operation: F, storage: bool) -> i32 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => {
            set_error("");
            0
        }
        Ok(Err(error)) => {
            set_error(&error.0);
            if storage {
                2
            } else {
                1
            }
        }
        Err(_) => {
            set_error("native hue operation panicked; no output was returned");
            3
        }
    }
}

fn error(message: &str) -> Error {
    Error(message.to_owned())
}

unsafe fn tensor<'a>(handle: *const Tensor) -> Result<&'a Tensor> {
    handle
        .as_ref()
        .ok_or_else(|| error("tensor handle is null"))
}

unsafe fn array<'a, T>(data: *const T, length: usize) -> Result<&'a [T]> {
    if length == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(error("input buffer is null"));
    }
    if length > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err(error("input buffer length overflows address space"));
    }
    Ok(std::slice::from_raw_parts(data, length))
}

unsafe fn shape(dims: *const usize, rank: usize) -> Result<Vec<usize>> {
    if rank > MAX_RANK {
        return Err(error("tensor rank exceeds 32"));
    }
    Ok(array(dims, rank)?.to_vec())
}

unsafe fn dimensions_for(dims: *const usize, rank: usize, count: usize) -> Result<Vec<usize>> {
    let dimensions = shape(dims, rank)?;
    if crate::shape_len(&dimensions)? != count {
        return Err(error("shape does not match the declared input token count"));
    }
    Ok(dimensions)
}

unsafe fn output<F: FnOnce() -> Result<Tensor>>(
    out: *mut *mut Tensor,
    operation: F,
    storage: bool,
) -> i32 {
    if out.is_null() {
        return boundary(|| Err(error("output handle pointer is null")), false);
    }
    *out = ptr::null_mut();
    boundary(
        || {
            let result = operation()?;
            *out = Box::into_raw(Box::new(result));
            Ok(())
        },
        storage,
    )
}

unsafe fn path<'a>(path: *const c_char) -> Result<&'a Path> {
    if path.is_null() {
        return Err(error("file path is null"));
    }
    let text = CStr::from_ptr(path)
        .to_str()
        .map_err(|_| error("file path must be UTF-8"))?;
    if text.is_empty() {
        return Err(error("file path is empty"));
    }
    Ok(Path::new(text))
}

#[no_mangle]
pub extern "C" fn ht_abi_version() -> u32 {
    ABI_VERSION
}
#[no_mangle]
pub extern "C" fn ht_modulus() -> u32 {
    MODULUS
}
#[no_mangle]
pub extern "C" fn ht_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

#[no_mangle]
/// # Safety
/// `phases` and `dims` must point to aligned readable arrays of `len` and
/// `rank` elements. `out` must point to writable handle storage. Input arrays
/// may be null when their length is zero. `out` must not overlap them.
pub unsafe extern "C" fn ht_new_phases(
    phases: *const u32,
    len: usize,
    dims: *const usize,
    rank: usize,
    out: *mut *mut Tensor,
) -> i32 {
    output(
        out,
        || {
            if len > MAX_ELEMENTS {
                return Err(error("phase buffer exceeds element budget"));
            }
            let dimensions = dimensions_for(dims, rank, len)?;
            Tensor::from_phases(array(phases, len)?.to_vec(), dimensions)
        },
        false,
    )
}

#[no_mangle]
/// # Safety
/// `bytes` must be readable for `nbytes` bytes and `dims` for `rank` aligned
/// dimensions. Zero-length inputs may be null. `out` must point to writable
/// handle storage that does not overlap either input.
pub unsafe extern "C" fn ht_new_rgbd(
    bytes: *const u8,
    nbytes: usize,
    dims: *const usize,
    rank: usize,
    out: *mut *mut Tensor,
) -> i32 {
    output(
        out,
        || {
            if !nbytes.is_multiple_of(4) || nbytes / 4 > MAX_ELEMENTS {
                return Err(error(
                    "RGBD buffer needs four bytes per token within the element budget",
                ));
            }
            let dimensions = dimensions_for(dims, rank, nbytes / 4)?;
            let bytes = array(bytes, nbytes)?;
            let mut tokens = Vec::new();
            tokens
                .try_reserve_exact(nbytes / 4)
                .map_err(|_| error("cannot allocate RGBD tokens"))?;
            for c in bytes.chunks_exact(4) {
                tokens.push(Token::new(c[0], c[1], c[2], c[3])?);
            }
            Tensor::from_rgbd(tokens, dimensions)
        },
        false,
    )
}

#[no_mangle]
/// # Safety
/// `bytes` must be readable for `nbytes` bytes and `dims` for `rank` aligned
/// dimensions. Zero-length inputs may be null. `out` must point to writable
/// handle storage that does not overlap either input.
pub unsafe extern "C" fn ht_new_rgb(
    bytes: *const u8,
    nbytes: usize,
    dims: *const usize,
    rank: usize,
    out: *mut *mut Tensor,
) -> i32 {
    output(
        out,
        || {
            if !nbytes.is_multiple_of(3) || nbytes / 3 > MAX_ELEMENTS {
                return Err(error(
                    "RGB buffer needs three bytes per color within the element budget",
                ));
            }
            let dimensions = dimensions_for(dims, rank, nbytes / 3)?;
            let bytes = array(bytes, nbytes)?;
            let mut tokens = Vec::new();
            tokens
                .try_reserve_exact(nbytes / 3)
                .map_err(|_| error("cannot allocate RGB hue tokens"))?;
            for c in bytes.chunks_exact(3) {
                tokens.push(Token::from_rgb(c[0], c[1], c[2])?);
            }
            Tensor::from_rgbd(tokens, dimensions)
        },
        false,
    )
}

#[no_mangle]
/// # Safety
/// A nonnull `handle` must be a live owned handle returned by this library.
/// Free it exactly once and only after every operation using it has ended.
pub unsafe extern "C" fn ht_free(handle: *mut Tensor) {
    if !handle.is_null() {
        let _ = boundary(
            || {
                drop(Box::from_raw(handle));
                Ok(())
            },
            false,
        );
    }
}

#[no_mangle]
/// # Safety
/// A nonnull `handle` must remain a valid live library handle during this call.
pub unsafe extern "C" fn ht_len(handle: *const Tensor) -> usize {
    match catch_unwind(AssertUnwindSafe(|| tensor(handle).map(Tensor::len))) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => {
            set_error(&e.0);
            0
        }
        Err(_) => {
            set_error("native count query panicked");
            0
        }
    }
}
#[no_mangle]
/// # Safety
/// A nonnull `handle` must remain a valid live library handle during this call.
pub unsafe extern "C" fn ht_ndim(handle: *const Tensor) -> usize {
    match catch_unwind(AssertUnwindSafe(|| tensor(handle).map(Tensor::ndim))) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => {
            set_error(&e.0);
            0
        }
        Err(_) => {
            set_error("native rank query panicked");
            0
        }
    }
}

#[no_mangle]
/// # Safety
/// `handle` must remain live during the call. `out` must be aligned and
/// writable for the tensor rank, with no overlap with the handle allocation.
/// A zero-rank output may be null.
pub unsafe extern "C" fn ht_copy_shape(
    handle: *const Tensor,
    out: *mut usize,
    capacity: usize,
) -> i32 {
    boundary(
        || {
            let shape = tensor(handle)?.shape();
            if capacity < shape.len() {
                return Err(error("shape output capacity is too small"));
            }
            if !shape.is_empty() {
                if out.is_null() {
                    return Err(error("shape output is null"));
                }
                ptr::copy_nonoverlapping(shape.as_ptr(), out, shape.len());
            }
            Ok(())
        },
        false,
    )
}

#[no_mangle]
/// # Safety
/// `handle` must remain live during the call. `out` must be writable for
/// four bytes per token, with no overlap with the handle allocation. An empty
/// tensor's output may be null.
pub unsafe extern "C" fn ht_copy_tokens(
    handle: *const Tensor,
    out: *mut u8,
    capacity: usize,
) -> i32 {
    boundary(
        || {
            let tokens = tensor(handle)?.tokens();
            let bytes = tokens.len() * 4;
            if capacity < bytes {
                return Err(error("RGBD output byte capacity is too small"));
            }
            if bytes != 0 {
                if out.is_null() {
                    return Err(error("RGBD output is null"));
                }
                ptr::copy_nonoverlapping(tokens.as_ptr().cast::<u8>(), out, bytes);
            }
            Ok(())
        },
        false,
    )
}

#[no_mangle]
/// # Safety
/// `handle` must remain live during the call. `out` must be writable for
/// three bytes per token, with no overlap with the handle allocation. An empty
/// tensor's output may be null.
pub unsafe extern "C" fn ht_copy_rgb(handle: *const Tensor, out: *mut u8, capacity: usize) -> i32 {
    boundary(
        || {
            let tokens = tensor(handle)?.tokens();
            let bytes = tokens.len() * 3;
            if capacity < bytes {
                return Err(error("RGB output byte capacity is too small"));
            }
            if bytes != 0 && out.is_null() {
                return Err(error("RGB output is null"));
            }
            for (index, token) in tokens.iter().enumerate() {
                *out.add(index * 3) = token.r;
                *out.add(index * 3 + 1) = token.g;
                *out.add(index * 3 + 2) = token.b;
            }
            Ok(())
        },
        false,
    )
}

#[no_mangle]
/// # Safety
/// `handle` must remain live during the call. `out` must be aligned and
/// writable for one u32 per token, with no overlap with the handle allocation.
/// An empty tensor's output may be null.
pub unsafe extern "C" fn ht_copy_phases(
    handle: *const Tensor,
    out: *mut u32,
    capacity: usize,
) -> i32 {
    boundary(
        || {
            let tokens = tensor(handle)?.tokens();
            if capacity < tokens.len() {
                return Err(error("phase output capacity is too small"));
            }
            if !tokens.is_empty() && out.is_null() {
                return Err(error("phase output is null"));
            }
            for (index, token) in tokens.iter().enumerate() {
                *out.add(index) = token.phase();
            }
            Ok(())
        },
        false,
    )
}

#[no_mangle]
/// # Safety
/// `a` and `b` must remain live library handles during the call. `out` must
/// point to writable handle storage outside either input allocation.
pub unsafe extern "C" fn ht_binary(
    a: *const Tensor,
    b: *const Tensor,
    opcode: u32,
    out: *mut *mut Tensor,
) -> i32 {
    output(
        out,
        || {
            let a = tensor(a)?;
            let b = tensor(b)?;
            match opcode {
                0 => a.add(b),
                1 => a.sub(b),
                2 => a.product(b),
                _ => Err(error("unknown binary operation code")),
            }
        },
        false,
    )
}
#[no_mangle]
/// # Safety
/// `a` must remain live during the call. `out` must point to writable handle
/// storage outside the input allocation.
pub unsafe extern "C" fn ht_rotate(a: *const Tensor, ticks: i64, out: *mut *mut Tensor) -> i32 {
    output(out, || tensor(a)?.rotate(ticks), false)
}
#[no_mangle]
/// # Safety
/// `a` and `b` must remain live library handles during the call. `out` must
/// point to writable handle storage outside either input allocation.
pub unsafe extern "C" fn ht_blend(
    a: *const Tensor,
    b: *const Tensor,
    weight: u8,
    out: *mut *mut Tensor,
) -> i32 {
    output(out, || tensor(a)?.blend(tensor(b)?, weight), false)
}
#[no_mangle]
/// # Safety
/// `a` and `b` must remain live library handles during the call. `out` must
/// point to writable handle storage outside either input allocation.
pub unsafe extern "C" fn ht_matmul(
    a: *const Tensor,
    b: *const Tensor,
    out: *mut *mut Tensor,
) -> i32 {
    output(out, || tensor(a)?.matmul(tensor(b)?), false)
}
#[no_mangle]
/// # Safety
/// `a` and `b` must remain live library handles during the call. `out` must
/// point to writable handle storage outside either input allocation.
pub unsafe extern "C" fn ht_dot(a: *const Tensor, b: *const Tensor, out: *mut *mut Tensor) -> i32 {
    output(out, || tensor(a)?.dot(tensor(b)?), false)
}
#[no_mangle]
/// # Safety
/// `a` must remain live during the call. `out` must point to writable handle
/// storage outside the input allocation.
pub unsafe extern "C" fn ht_sum(a: *const Tensor, axis: usize, out: *mut *mut Tensor) -> i32 {
    output(out, || tensor(a)?.sum(axis), false)
}
#[no_mangle]
/// # Safety
/// `a` must remain live during the call. `dims` must be readable and aligned
/// for `rank` dimensions, or null for zero rank. `out` must point to writable
/// handle storage outside the input allocations.
pub unsafe extern "C" fn ht_reshape(
    a: *const Tensor,
    dims: *const usize,
    rank: usize,
    out: *mut *mut Tensor,
) -> i32 {
    output(out, || tensor(a)?.reshape(shape(dims, rank)?), false)
}
#[no_mangle]
/// # Safety
/// `a` must remain live during the call. `axes` must be readable and aligned
/// for `rank` entries, or null for zero rank. `out` must point to writable
/// handle storage outside the input allocations.
pub unsafe extern "C" fn ht_permute(
    a: *const Tensor,
    axes: *const usize,
    rank: usize,
    out: *mut *mut Tensor,
) -> i32 {
    output(
        out,
        || {
            if rank > MAX_RANK {
                return Err(error("permutation rank exceeds 32"));
            }
            tensor(a)?.permute(array(axes, rank)?)
        },
        false,
    )
}
#[no_mangle]
/// # Safety
/// `a` must remain live during the call. `out` must point to writable handle
/// storage outside the input allocation.
pub unsafe extern "C" fn ht_slice(
    a: *const Tensor,
    axis: usize,
    start: usize,
    stop: usize,
    step: usize,
    out: *mut *mut Tensor,
) -> i32 {
    output(out, || tensor(a)?.slice(axis, start, stop, step), false)
}
#[no_mangle]
/// # Safety
/// `a` must remain live during the call. `filename` must point to a readable
/// NUL-terminated UTF-8 string during the call.
pub unsafe extern "C" fn ht_save(a: *const Tensor, filename: *const c_char) -> i32 {
    boundary(
        || crate::storage::save_tensor(tensor(a)?, path(filename)?),
        true,
    )
}
#[no_mangle]
/// # Safety
/// `filename` must point to a readable NUL-terminated UTF-8 string during the
/// call. `out` must point to writable handle storage outside that string.
pub unsafe extern "C" fn ht_load(
    filename: *const c_char,
    max_bytes: usize,
    out: *mut *mut Tensor,
) -> i32 {
    output(
        out,
        || crate::storage::load_tensor(path(filename)?, max_bytes),
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_shape_before_reading_or_allocating_input_buffers() {
        unsafe {
            let dimensions = [1usize];
            let unreadable_phase = std::ptr::NonNull::<u32>::dangling().as_ptr();
            let unreadable_color = std::ptr::NonNull::<u8>::dangling().as_ptr();
            let mut out = ptr::dangling_mut::<Tensor>();
            assert_eq!(
                ht_new_phases(
                    unreadable_phase,
                    MAX_ELEMENTS,
                    dimensions.as_ptr(),
                    1,
                    &mut out
                ),
                1
            );
            assert!(out.is_null());
            assert!(CStr::from_ptr(ht_last_error())
                .to_str()
                .unwrap()
                .contains("shape"));
            assert_eq!(
                ht_new_rgbd(
                    unreadable_color,
                    MAX_ELEMENTS * 4,
                    dimensions.as_ptr(),
                    1,
                    &mut out
                ),
                1
            );
            assert!(out.is_null());
            assert_eq!(
                ht_new_rgb(
                    unreadable_color,
                    MAX_ELEMENTS * 3,
                    dimensions.as_ptr(),
                    1,
                    &mut out
                ),
                1
            );
            assert!(out.is_null());
            let excessive_dimension = [0, MAX_ELEMENTS + 1];
            assert_eq!(
                ht_new_phases(ptr::null(), 0, excessive_dimension.as_ptr(), 2, &mut out),
                1
            );
            assert!(out.is_null());
        }
    }
    #[test]
    fn abi_nulls_capacities_and_output_error_contract() {
        unsafe {
            assert_eq!(ht_abi_version(), 1);
            assert_eq!(ht_modulus(), MODULUS);
            let mut out = ptr::dangling_mut::<Tensor>();
            assert_eq!(ht_new_phases(ptr::null(), 1, ptr::null(), 0, &mut out), 1);
            assert!(out.is_null());
            assert!(!CStr::from_ptr(ht_last_error()).to_bytes().is_empty());
            assert_eq!(
                ht_new_phases(ptr::null(), 0, ptr::null(), 0, ptr::null_mut()),
                1
            );
            let phase = [5u32];
            assert_eq!(
                ht_new_phases(phase.as_ptr(), 1, ptr::null(), 0, &mut out),
                0
            );
            assert_eq!(ht_len(out), 1);
            assert_eq!(ht_ndim(out), 0);
            assert_eq!(ht_copy_shape(out, ptr::null_mut(), 0), 0);
            assert_eq!(ht_copy_tokens(out, ptr::null_mut(), 3), 1);
            let mut bytes = [0; 4];
            assert_eq!(ht_copy_tokens(out, bytes.as_mut_ptr(), 4), 0);
            assert_eq!(bytes, [255, 0, 0, 5]);
            let mut rgb = [0; 3];
            assert_eq!(ht_copy_rgb(out, rgb.as_mut_ptr(), 3), 0);
            assert_eq!(rgb, [255, 0, 0]);
            assert_eq!(ht_copy_rgb(out, ptr::null_mut(), 2), 1);
            let mut q = [0];
            assert_eq!(ht_copy_phases(out, q.as_mut_ptr(), 1), 0);
            assert_eq!(q, [5]);
            let mut result = ptr::dangling_mut::<Tensor>();
            assert_eq!(ht_binary(out, out, 99, &mut result), 1);
            assert!(result.is_null());
            assert_eq!(ht_rotate(out, 1, &mut result), 0);
            assert_eq!(ht_copy_phases(result, q.as_mut_ptr(), 1), 0);
            assert_eq!(q, [6]);
            ht_free(result);
            ht_free(out);
            ht_free(ptr::null_mut());
        }
    }

    #[test]
    fn panic_is_caught_and_last_error_is_thread_local() {
        assert_eq!(
            boundary(
                || -> Result<()> {
                    panic!("test panic");
                },
                false
            ),
            3
        );
        unsafe {
            assert!(CStr::from_ptr(ht_last_error())
                .to_str()
                .unwrap()
                .contains("panicked"));
        }
        let worker = std::thread::spawn(|| {
            set_error("worker");
            unsafe { CStr::from_ptr(ht_last_error()).to_str().unwrap().to_owned() }
        });
        assert_eq!(worker.join().unwrap(), "worker");
        unsafe {
            assert!(CStr::from_ptr(ht_last_error())
                .to_str()
                .unwrap()
                .contains("panicked"));
        }
    }
}
