//! Portable persistence of the actual RGBD token bytes.
//!
//! Version 1 uses little-endian lengths and a SHA-256 footer. The checksum
//! covers the header, shape and every token byte. It detects corruption; it
//! does not authenticate the sender of a file.

use crate::{Error, Result, Tensor, Token, MAX_ELEMENTS, MAX_RANK};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC: [u8; 8] = *b"HUECT\0\r\n";
const VERSION: u16 = 1;
const FIXED_HEADER: usize = 24;
const CHECKSUM_BYTES: usize = 32;
const TOKEN_BYTES: usize = 4;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn failure(message: impl Into<String>) -> Error {
    Error(message.into())
}

fn io_failure(action: &str, error: std::io::Error) -> Error {
    failure(format!("{action}: {error}"))
}

fn dimensions_count(shape: &[usize]) -> Result<usize> {
    if shape.len() > MAX_RANK {
        return Err(failure(format!("tensor rank exceeds {MAX_RANK}")));
    }
    let count = shape.iter().try_fold(1usize, |total, &dimension| {
        total
            .checked_mul(dimension)
            .ok_or_else(|| failure("tensor shape overflows the address space"))
    })?;
    if count > MAX_ELEMENTS {
        return Err(failure("tensor exceeds the element allocation budget"));
    }
    Ok(count)
}

fn expected_bytes(rank: usize, token_count: usize) -> Result<usize> {
    rank.checked_mul(8)
        .and_then(|shape_bytes| FIXED_HEADER.checked_add(shape_bytes))
        .and_then(|header_bytes| {
            token_count
                .checked_mul(TOKEN_BYTES)
                .and_then(|payload_bytes| header_bytes.checked_add(payload_bytes))
        })
        .and_then(|body_bytes| body_bytes.checked_add(CHECKSUM_BYTES))
        .ok_or_else(|| failure("tensor file length overflows the address space"))
}

fn adjacent_temporary(path: &Path) -> Result<(PathBuf, File)> {
    let name = path
        .file_name()
        .ok_or_else(|| failure("tensor output path must name a file"))?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let instant = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| failure("system clock precedes the Unix epoch"))?
        .as_nanos();
    for _ in 0..16 {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = OsString::from(".");
        temporary_name.push(name);
        temporary_name.push(format!(
            ".{}.{}.{}.tmp",
            std::process::id(),
            instant,
            counter
        ));
        let temporary_path = parent.join(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_failure("create adjacent tensor temporary file", error)),
        }
    }
    Err(failure("could not create a unique tensor temporary file"))
}

fn write_hashed(file: &mut File, hash: &mut Sha256, bytes: &[u8]) -> Result<()> {
    file.write_all(bytes)
        .map_err(|error| io_failure("write tensor file", error))?;
    hash.update(bytes);
    Ok(())
}

/// Save the stored color tokens, without an auxiliary numerical tensor.
///
/// The writer creates a new temporary file in the destination directory,
/// flushes it to disk, then renames it to the requested path. An unsuccessful
/// write leaves an existing destination unchanged. The final rename follows
/// the platform's filesystem semantics.
pub fn save_tensor(tensor: &Tensor, path: &Path) -> Result<()> {
    let count = dimensions_count(tensor.shape())?;
    if count != tensor.tokens().len() {
        return Err(failure("tensor shape does not match its token count"));
    }
    expected_bytes(tensor.shape().len(), count)?;
    let count_u64 = u64::try_from(count).map_err(|_| failure("token count exceeds u64"))?;
    let rank_u32 = u32::try_from(tensor.shape().len()).map_err(|_| failure("rank exceeds u32"))?;
    let (temporary_path, mut file) = adjacent_temporary(path)?;
    let write_result = (|| -> Result<()> {
        let mut hash = Sha256::new();
        let mut header = [0u8; FIXED_HEADER];
        header[..8].copy_from_slice(&MAGIC);
        header[8..10].copy_from_slice(&VERSION.to_le_bytes());
        // Bytes 10..12 are reserved flags and must remain zero in version 1.
        header[12..16].copy_from_slice(&rank_u32.to_le_bytes());
        header[16..24].copy_from_slice(&count_u64.to_le_bytes());
        write_hashed(&mut file, &mut hash, &header)?;
        for &dimension in tensor.shape() {
            let dimension_u64 =
                u64::try_from(dimension).map_err(|_| failure("dimension exceeds u64"))?;
            write_hashed(&mut file, &mut hash, &dimension_u64.to_le_bytes())?;
        }
        let mut chunk = [0u8; 4096 * TOKEN_BYTES];
        for tokens in tensor.tokens().chunks(4096) {
            for (index, token) in tokens.iter().enumerate() {
                let start = index * TOKEN_BYTES;
                chunk[start..start + TOKEN_BYTES]
                    .copy_from_slice(&[token.r, token.g, token.b, token.d]);
            }
            write_hashed(&mut file, &mut hash, &chunk[..tokens.len() * TOKEN_BYTES])?;
        }
        file.write_all(&hash.finish())
            .map_err(|error| io_failure("write tensor checksum", error))?;
        file.sync_all()
            .map_err(|error| io_failure("sync tensor file", error))?;
        Ok(())
    })();
    drop(file);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary_path);
        return Err(error);
    }
    match fs::rename(&temporary_path, path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = fs::remove_file(&temporary_path);
            Err(io_failure("replace tensor file", error))
        }
    }
}

fn read_hashed(file: &mut File, hash: &mut Sha256, bytes: &mut [u8]) -> Result<()> {
    file.read_exact(bytes)
        .map_err(|error| io_failure("read tensor file", error))?;
    hash.update(bytes);
    Ok(())
}

/// Load actual RGBD tokens, checking the file size before allocating tokens.
///
/// `max_bytes` limits the complete on-disk file, including metadata and the
/// checksum. Rank, shape, token count, reserved fields and trailing bytes are
/// checked before the constructor validates canonical colors.
pub fn load_tensor(path: &Path, max_bytes: usize) -> Result<Tensor> {
    let mut file = File::open(path).map_err(|error| io_failure("open tensor file", error))?;
    let metadata = file
        .metadata()
        .map_err(|error| io_failure("read tensor file metadata", error))?;
    if !metadata.is_file() {
        return Err(failure("tensor input must be a regular file"));
    }
    let budget = u64::try_from(max_bytes).unwrap_or(u64::MAX);
    if metadata.len() > budget {
        return Err(failure(format!(
            "tensor file contains {} bytes and exceeds the {max_bytes} byte budget",
            metadata.len()
        )));
    }
    if metadata.len() < (FIXED_HEADER + CHECKSUM_BYTES) as u64 {
        return Err(failure(
            "tensor file is shorter than its fixed header and checksum",
        ));
    }
    let file_bytes = usize::try_from(metadata.len())
        .map_err(|_| failure("tensor file exceeds the address space"))?;
    let mut hash = Sha256::new();
    let mut header = [0u8; FIXED_HEADER];
    read_hashed(&mut file, &mut hash, &mut header)?;
    if header[..8] != MAGIC {
        return Err(failure("unrecognized hue tensor file magic"));
    }
    let version = u16::from_le_bytes([header[8], header[9]]);
    if version != VERSION {
        return Err(failure(format!(
            "unsupported hue tensor file version {version}"
        )));
    }
    if header[10] != 0 || header[11] != 0 {
        return Err(failure("reserved hue tensor file flags must be zero"));
    }
    let rank = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    if rank > MAX_RANK {
        return Err(failure(format!("tensor rank exceeds {MAX_RANK}")));
    }
    let count = usize::try_from(u64::from_le_bytes(header[16..24].try_into().unwrap()))
        .map_err(|_| failure("token count exceeds the address space"))?;
    if count > MAX_ELEMENTS {
        return Err(failure("tensor exceeds the element allocation budget"));
    }
    if expected_bytes(rank, count)? != file_bytes {
        return Err(failure(
            "tensor file length does not match its rank and token count",
        ));
    }
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        let mut dimension_bytes = [0u8; 8];
        read_hashed(&mut file, &mut hash, &mut dimension_bytes)?;
        shape.push(
            usize::try_from(u64::from_le_bytes(dimension_bytes))
                .map_err(|_| failure("tensor dimension exceeds the address space"))?,
        );
    }
    if dimensions_count(&shape)? != count {
        return Err(failure("tensor shape does not match its token count"));
    }
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(count)
        .map_err(|_| failure("cannot allocate the tensor token buffer within the address space"))?;
    let mut chunk = [0u8; 4096 * TOKEN_BYTES];
    let mut remaining = count;
    while remaining > 0 {
        let block_count = remaining.min(4096);
        let payload = &mut chunk[..block_count * TOKEN_BYTES];
        read_hashed(&mut file, &mut hash, payload)?;
        for bytes in payload.chunks_exact(TOKEN_BYTES) {
            tokens.push(Token {
                r: bytes[0],
                g: bytes[1],
                b: bytes[2],
                d: bytes[3],
            });
        }
        remaining -= block_count;
    }
    let mut stored_checksum = [0u8; CHECKSUM_BYTES];
    file.read_exact(&mut stored_checksum)
        .map_err(|error| io_failure("read tensor checksum", error))?;
    if hash.finish() != stored_checksum {
        return Err(failure("hue tensor checksum mismatch"));
    }
    let mut trailing_byte = [0u8; 1];
    if file
        .read(&mut trailing_byte)
        .map_err(|error| io_failure("check tensor trailing bytes", error))?
        != 0
    {
        return Err(failure("tensor file contains trailing bytes"));
    }
    Tensor::from_rgbd(tokens, shape)
}

// A small streaming SHA-256 implementation keeps the file format independent
// of a package manager or platform crypto service. Arithmetic is explicitly
// wrapping because SHA-256 operates in the ring of 32-bit words.
struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    byte_count: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0u8; 64],
            buffered: 0,
            byte_count: 0,
        }
    }

    fn update(&mut self, mut bytes: &[u8]) {
        self.byte_count = self.byte_count.wrapping_add(bytes.len() as u64);
        if self.buffered != 0 {
            let taken = (64 - self.buffered).min(bytes.len());
            self.buffer[self.buffered..self.buffered + taken].copy_from_slice(&bytes[..taken]);
            self.buffered += taken;
            bytes = &bytes[taken..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while bytes.len() >= 64 {
            let block: &[u8; 64] = bytes[..64].try_into().unwrap();
            self.compress(block);
            bytes = &bytes[64..];
        }
        if !bytes.is_empty() {
            self.buffer[..bytes.len()].copy_from_slice(bytes);
            self.buffered = bytes.len();
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut schedule = [0u32; 64];
        for (index, word) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for index in 16..64 {
            let x = schedule[index - 15];
            let y = schedule[index - 2];
            let small0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
            let small1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(small0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(small1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let large1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(large1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let large0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = large0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (old, added) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *old = old.wrapping_add(added);
        }
    }

    fn finish(mut self) -> [u8; CHECKSUM_BYTES] {
        let bit_count = self.byte_count.wrapping_mul(8);
        self.buffer[self.buffered] = 0x80;
        self.buffered += 1;
        if self.buffered > 56 {
            self.buffer[self.buffered..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffered = 0;
        }
        self.buffer[self.buffered..56].fill(0);
        self.buffer[56..64].copy_from_slice(&bit_count.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        let mut result = [0u8; CHECKSUM_BYTES];
        for (index, word) in self.state.iter().enumerate() {
            result[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPath(PathBuf);

    impl TestPath {
        fn new() -> Self {
            let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let instant = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(std::env::temp_dir().join(format!(
                "hue-storage-test-{}-{instant}-{counter}.hue",
                std::process::id()
            )))
        }
    }

    impl Drop for TestPath {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn fixture() -> Tensor {
        Tensor::from_phases(vec![0, 256, 131_072, 391_679], vec![2, 2]).unwrap()
    }

    fn checksum(bytes: &mut [u8]) {
        let footer = bytes.len() - CHECKSUM_BYTES;
        let mut hash = Sha256::new();
        hash.update(&bytes[..footer]);
        bytes[footer..].copy_from_slice(&hash.finish());
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn sha256_matches_independent_known_vectors() {
        let mut empty = Sha256::new();
        empty.update(&[]);
        assert_eq!(
            hex(&empty.finish()),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let mut abc = Sha256::new();
        abc.update(b"abc");
        assert_eq!(
            hex(&abc.finish()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut boundary = Sha256::new();
        boundary.update(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            hex(&boundary.finish()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        let mut million = Sha256::new();
        let block = [b'a'; 1000];
        for _ in 0..1000 {
            million.update(&block);
        }
        assert_eq!(
            hex(&million.finish()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn sha256_streaming_matches_single_write_across_boundaries() {
        let bytes: Vec<u8> = (0..1025).map(|index| (index % 256) as u8).collect();
        let mut whole = Sha256::new();
        whole.update(&bytes);
        let expected = whole.finish();
        for width in [1, 2, 7, 55, 56, 63, 64, 65, 127] {
            let mut pieces = Sha256::new();
            for chunk in bytes.chunks(width) {
                pieces.update(chunk);
            }
            assert_eq!(pieces.finish(), expected, "chunk size {width}");
        }
    }

    #[test]
    fn persists_exact_color_bytes_with_portable_lengths() {
        let path = TestPath::new();
        let original = fixture();
        save_tensor(&original, &path.0).unwrap();
        let bytes = fs::read(&path.0).unwrap();
        assert_eq!(bytes.len(), 88);
        assert_eq!(&bytes[..8], &MAGIC);
        assert_eq!(&bytes[8..10], &[1, 0]);
        assert_eq!(&bytes[12..16], &[2, 0, 0, 0]);
        assert_eq!(&bytes[16..24], &[4, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&bytes[24..32], &[2, 0, 0, 0, 0, 0, 0, 0]);
        let actual: Vec<u8> = original
            .tokens()
            .iter()
            .flat_map(|token| [token.r, token.g, token.b, token.d])
            .collect();
        assert_eq!(&bytes[40..56], actual.as_slice());
        let loaded = load_tensor(&path.0, bytes.len()).unwrap();
        assert_eq!(loaded.shape(), original.shape());
        assert_eq!(loaded.tokens(), original.tokens());
        assert_eq!(loaded.phases(), original.phases());
    }

    #[test]
    fn handles_scalar_empty_and_multiple_payload_blocks() {
        for (phases, shape) in [
            (vec![390_000], vec![]),
            (vec![], vec![0, 3]),
            ((0..5000).collect(), vec![100, 50]),
        ] {
            let path = TestPath::new();
            let original = Tensor::from_phases(phases, shape).unwrap();
            save_tensor(&original, &path.0).unwrap();
            let loaded = load_tensor(&path.0, 1_000_000).unwrap();
            assert_eq!(loaded.shape(), original.shape());
            assert_eq!(loaded.tokens(), original.tokens());
        }
    }

    #[test]
    fn rejects_size_budget_before_payload_allocation() {
        let path = TestPath::new();
        save_tensor(&fixture(), &path.0).unwrap();
        let length = fs::metadata(&path.0).unwrap().len() as usize;
        assert!(load_tensor(&path.0, length - 1)
            .unwrap_err()
            .0
            .contains("budget"));
        assert!(load_tensor(&path.0, 0).unwrap_err().0.contains("budget"));
    }

    #[test]
    fn detects_every_payload_channel_mutation() {
        let path = TestPath::new();
        save_tensor(&fixture(), &path.0).unwrap();
        let original = fs::read(&path.0).unwrap();
        for channel in 0..TOKEN_BYTES {
            let mut changed = original.clone();
            changed[40 + channel] ^= 1;
            fs::write(&path.0, &changed).unwrap();
            assert!(load_tensor(&path.0, 1024)
                .unwrap_err()
                .0
                .contains("checksum"));
        }
    }

    #[test]
    fn rejects_noncanonical_color_even_with_valid_checksum() {
        let path = TestPath::new();
        save_tensor(&fixture(), &path.0).unwrap();
        let mut changed = fs::read(&path.0).unwrap();
        changed[40..44].copy_from_slice(&[1, 2, 3, 4]);
        checksum(&mut changed);
        fs::write(&path.0, changed).unwrap();
        assert!(load_tensor(&path.0, 1024).is_err());
    }

    #[test]
    fn changed_d_byte_survives_persistence_and_changes_decoding() {
        let path = TestPath::new();
        save_tensor(&fixture(), &path.0).unwrap();
        let mut changed = fs::read(&path.0).unwrap();
        changed[43] = 9;
        checksum(&mut changed);
        fs::write(&path.0, changed).unwrap();
        let loaded = load_tensor(&path.0, 1024).unwrap();
        assert_eq!(loaded.tokens()[0].d, 9);
        assert_eq!(loaded.phases()[0], 9);
    }

    #[test]
    fn rejects_bad_magic_version_flags_rank_count_and_shape() {
        let path = TestPath::new();
        save_tensor(&fixture(), &path.0).unwrap();
        let original = fs::read(&path.0).unwrap();
        for (offset, value, message) in [
            (0, 0, "magic"),
            (8, 2, "version"),
            (10, 1, "flags"),
            (12, 33, "rank"),
            (16, 5, "length"),
            (24, 3, "shape"),
        ] {
            let mut changed = original.clone();
            changed[offset] = value;
            checksum(&mut changed);
            fs::write(&path.0, changed).unwrap();
            assert!(
                load_tensor(&path.0, 1024).unwrap_err().0.contains(message),
                "{message}"
            );
        }
    }

    #[test]
    fn rejects_truncation_trailing_bytes_and_short_files() {
        let path = TestPath::new();
        save_tensor(&fixture(), &path.0).unwrap();
        let original = fs::read(&path.0).unwrap();
        for length in [0, 23, 55, original.len() - 1] {
            fs::write(&path.0, &original[..length]).unwrap();
            assert!(load_tensor(&path.0, 1024).is_err());
        }
        let mut longer = original;
        longer.push(0);
        fs::write(&path.0, longer).unwrap();
        assert!(load_tensor(&path.0, 1024).unwrap_err().0.contains("length"));
    }

    #[test]
    fn overwrites_existing_file_without_leaking_temporary_files() {
        let path = TestPath::new();
        fs::write(&path.0, b"old contents").unwrap();
        save_tensor(&fixture(), &path.0).unwrap();
        assert_eq!(
            load_tensor(&path.0, 1024).unwrap().phases(),
            fixture().phases()
        );
    }
}
