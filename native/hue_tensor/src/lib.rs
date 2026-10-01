//! Actual RGBD symbols on a finite, discrete hue circle.
//!
//! The three color channels identify one of 1,530 saturated RGB colors.
//! D divides the step to the next color into 256 subdivisions. The domain
//! is the ring of phase ticks modulo 391,680. No real-valued numerical
//! tensors exist inside this library.

use std::fmt;

pub mod ffi;
pub mod storage;

pub const ABI_VERSION: u32 = 1;
pub const HUE_STEPS: u32 = 1530;
pub const MODULUS: u32 = HUE_STEPS * 256;
pub const MAX_RANK: usize = 32;
pub const MAX_ELEMENTS: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

fn invalid(message: &str) -> Error {
    Error(message.to_owned())
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Token {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub d: u8,
}

impl Token {
    pub fn new(r: u8, g: u8, b: u8, d: u8) -> Result<Self> {
        let token = Self { r, g, b, d };
        token.hue()?;
        Ok(token)
    }

    /// Encode a phase tick into actual color bytes. Out-of-domain ticks fail.
    pub fn from_phase(phase: u32) -> Result<Self> {
        if phase >= MODULUS {
            return Err(invalid("phase tick is outside the hue domain"));
        }
        Ok(Self::from_phase_valid(phase))
    }

    fn from_phase_valid(phase: u32) -> Self {
        let hue = phase / 256;
        let sector = hue / 255;
        let t = (hue % 255) as u8;
        let (r, g, b) = match sector {
            0 => (255, t, 0),
            1 => (255 - t, 255, 0),
            2 => (0, 255, t),
            3 => (0, 255 - t, 255),
            4 => (t, 0, 255),
            _ => (255, 0, 255 - t),
        };
        Self {
            r,
            g,
            b,
            d: (phase % 256) as u8,
        }
    }

    fn hue(&self) -> Result<u32> {
        let Self { r, g, b, .. } = *self;
        let hue = if r == 255 && b == 0 && g < 255 {
            u32::from(g)
        } else if g == 255 && b == 0 && r > 0 {
            255 + u32::from(255 - r)
        } else if r == 0 && g == 255 && b < 255 {
            510 + u32::from(b)
        } else if r == 0 && b == 255 && g > 0 {
            765 + u32::from(255 - g)
        } else if g == 0 && b == 255 && r < 255 {
            1020 + u32::from(r)
        } else if r == 255 && g == 0 && b > 0 {
            1275 + u32::from(255 - b)
        } else {
            return Err(invalid("RGBD requires a canonical saturated hue color"));
        };
        Ok(hue)
    }

    /// Integer phase reconstruction. Validated tokens are the tensor storage.
    pub fn phase(&self) -> u32 {
        self.hue().expect("Token must have canonical RGB channels") * 256 + u32::from(self.d)
    }

    /// Project RGB to hue using integer channel differences. Saturation and
    /// brightness are discarded. An achromatic color has no hue and fails.
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Result<Self> {
        let r = i64::from(r);
        let g = i64::from(g);
        let b = i64::from(b);
        let maximum = r.max(g).max(b);
        let minimum = r.min(g).min(b);
        let span = maximum - minimum;
        if span == 0 {
            return Err(invalid("achromatic RGB has no hue"));
        }
        // Hue numerator is in sixth-turn units. Euclidean remainder puts
        // red across the circle seam without a signed rounding ambiguity.
        let numerator = if maximum == r {
            g - b
        } else if maximum == g {
            2 * span + b - r
        } else {
            4 * span + r - g
        };
        let numerator = numerator.rem_euclid(6 * span);
        let denominator = 6 * span;
        let rounded = (numerator * i64::from(MODULUS) + denominator / 2) / denominator;
        Ok(Self::from_phase_valid(
            (rounded % i64::from(MODULUS)) as u32,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub(crate) tokens: Vec<Token>,
    pub(crate) shape: Vec<usize>,
}

fn shape_len(shape: &[usize]) -> Result<usize> {
    if shape.len() > MAX_RANK {
        return Err(invalid("tensor rank exceeds 32"));
    }
    let mut length = 1usize;
    for &dimension in shape {
        if dimension > MAX_ELEMENTS {
            return Err(invalid(
                "tensor dimension exceeds the element allocation budget",
            ));
        }
        length = length
            .checked_mul(dimension)
            .ok_or_else(|| invalid("tensor shape overflows address space"))?;
    }
    if length > MAX_ELEMENTS {
        return Err(invalid("tensor exceeds the element allocation budget"));
    }
    Ok(length)
}

fn allocate_tokens(length: usize) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(length)
        .map_err(|_| invalid("cannot allocate hue tensor storage"))?;
    Ok(tokens)
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1usize;
    let mut result = vec![0; shape.len()];
    for index in (0..shape.len()).rev() {
        result[index] = stride;
        stride = stride.saturating_mul(shape[index]);
    }
    result
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1; rank];
    for i in 0..rank {
        let x = a.len().checked_sub(i + 1).map(|j| a[j]).unwrap_or(1);
        let y = b.len().checked_sub(i + 1).map(|j| b[j]).unwrap_or(1);
        out[rank - 1 - i] = if x == y {
            x
        } else if x == 1 {
            y
        } else if y == 1 {
            x
        } else {
            return Err(invalid("tensor shapes do not broadcast"));
        };
    }
    shape_len(&out)?;
    Ok(out)
}

fn broadcast_offset(
    mut flat: usize,
    output: &[usize],
    input: &[usize],
    input_strides: &[usize],
) -> usize {
    let lead = output.len() - input.len();
    let mut offset = 0;
    for axis in (0..output.len()).rev() {
        let coordinate = flat % output[axis];
        flat /= output[axis];
        if axis >= lead && input[axis - lead] != 1 {
            offset += coordinate * input_strides[axis - lead];
        }
    }
    offset
}

impl Tensor {
    pub fn from_rgbd(tokens: Vec<Token>, shape: Vec<usize>) -> Result<Self> {
        if shape_len(&shape)? != tokens.len() {
            return Err(invalid("shape does not match color token count"));
        }
        for token in &tokens {
            token.hue()?;
        }
        Ok(Self { tokens, shape })
    }

    pub fn from_phases(phases: Vec<u32>, shape: Vec<usize>) -> Result<Self> {
        if shape_len(&shape)? != phases.len() {
            return Err(invalid("shape does not match phase count"));
        }
        let mut tokens = allocate_tokens(phases.len())?;
        for phase in phases {
            tokens.push(Token::from_phase(phase)?);
        }
        Ok(Self { tokens, shape })
    }

    pub fn from_rgb(rgb: &[[u8; 3]], shape: Vec<usize>) -> Result<Self> {
        if shape_len(&shape)? != rgb.len() {
            return Err(invalid("shape does not match RGB color count"));
        }
        let mut tokens = allocate_tokens(rgb.len())?;
        for &[r, g, b] in rgb {
            tokens.push(Token::from_rgb(r, g, b)?);
        }
        Ok(Self { tokens, shape })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn phases(&self) -> Vec<u32> {
        self.tokens.iter().map(Token::phase).collect()
    }

    fn binary<F: Fn(u32, u32) -> u32>(&self, other: &Self, operation: F) -> Result<Self> {
        let shape = broadcast_shape(&self.shape, &other.shape)?;
        let count = shape_len(&shape)?;
        let left_strides = strides(&self.shape);
        let right_strides = strides(&other.shape);
        let mut tokens = allocate_tokens(count)?;
        for flat in 0..count {
            let a = self.tokens[broadcast_offset(flat, &shape, &self.shape, &left_strides)].phase();
            let b =
                other.tokens[broadcast_offset(flat, &shape, &other.shape, &right_strides)].phase();
            tokens.push(Token::from_phase_valid(operation(a, b)));
        }
        Ok(Self { tokens, shape })
    }

    /// Composition follows phase addition around the hue circle.
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.binary(other, |a, b| (a + b) % MODULUS)
    }
    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.binary(other, |a, b| (a + MODULUS - b) % MODULUS)
    }
    pub fn product(&self, other: &Self) -> Result<Self> {
        self.binary(other, |a, b| {
            ((u64::from(a) * u64::from(b)) % u64::from(MODULUS)) as u32
        })
    }

    pub fn rotate(&self, ticks: i64) -> Result<Self> {
        let delta = ticks.rem_euclid(i64::from(MODULUS)) as u32;
        let mut tokens = allocate_tokens(self.len())?;
        for token in &self.tokens {
            tokens.push(Token::from_phase_valid((token.phase() + delta) % MODULUS));
        }
        Ok(Self {
            tokens,
            shape: self.shape.clone(),
        })
    }

    /// Move along the shortest arc. Weight 0 keeps the left color and 255
    /// reaches the right color. An exact half-circle takes the positive arc.
    /// Integer rounding uses nearest tick, with ties away from zero.
    pub fn blend(&self, other: &Self, weight: u8) -> Result<Self> {
        self.binary(other, |a, b| {
            let forward = (b + MODULUS - a) % MODULUS;
            let delta = if forward > MODULUS / 2 {
                i64::from(forward) - i64::from(MODULUS)
            } else {
                i64::from(forward)
            };
            let numerator = delta * i64::from(weight);
            let step = if numerator < 0 {
                -((-numerator + 127) / 255)
            } else {
                (numerator + 127) / 255
            };
            (i64::from(a) + step).rem_euclid(i64::from(MODULUS)) as u32
        })
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(invalid("matmul requires two matrices"));
        }
        let (rows, inner, columns) = (self.shape[0], self.shape[1], other.shape[1]);
        if inner != other.shape[0] {
            return Err(invalid("matmul contraction dimensions differ"));
        }
        let shape = vec![rows, columns];
        let count = shape_len(&shape)?;
        let mut tokens = allocate_tokens(count)?;
        for row in 0..rows {
            for column in 0..columns {
                let mut phase = 0u64;
                for k in 0..inner {
                    phase = (phase
                        + u64::from(self.tokens[row * inner + k].phase())
                            * u64::from(other.tokens[k * columns + column].phase()))
                        % u64::from(MODULUS);
                }
                tokens.push(Token::from_phase_valid(phase as u32));
            }
        }
        Ok(Self { tokens, shape })
    }

    pub fn dot(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(invalid("dot requires identical tensor shapes"));
        }
        let mut phase = 0u64;
        for (a, b) in self.tokens.iter().zip(&other.tokens) {
            phase = (phase + u64::from(a.phase()) * u64::from(b.phase())) % u64::from(MODULUS);
        }
        Ok(Self {
            tokens: vec![Token::from_phase_valid(phase as u32)],
            shape: vec![],
        })
    }

    pub fn sum(&self, axis: usize) -> Result<Self> {
        if axis >= self.ndim() {
            return Err(invalid("sum axis is outside the tensor"));
        }
        let mut shape = self.shape.clone();
        let dimension = shape.remove(axis);
        let count = shape_len(&shape)?;
        let input_strides = strides(&self.shape);
        let mut tokens = allocate_tokens(count)?;
        for flat in 0..count {
            let mut quotient = flat;
            let mut base = 0;
            for out_axis in (0..shape.len()).rev() {
                let coordinate = quotient % shape[out_axis];
                quotient /= shape[out_axis];
                let in_axis = if out_axis >= axis {
                    out_axis + 1
                } else {
                    out_axis
                };
                base += coordinate * input_strides[in_axis];
            }
            let mut phase = 0;
            for k in 0..dimension {
                phase = (phase + self.tokens[base + k * input_strides[axis]].phase()) % MODULUS;
            }
            tokens.push(Token::from_phase_valid(phase));
        }
        Ok(Self { tokens, shape })
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        if shape_len(&shape)? != self.len() {
            return Err(invalid("reshape must preserve token count"));
        }
        Ok(Self {
            tokens: self.tokens.clone(),
            shape,
        })
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.ndim() {
            return Err(invalid("permutation rank differs from tensor rank"));
        }
        let mut seen = vec![false; self.ndim()];
        for &axis in axes {
            if axis >= self.ndim() || seen[axis] {
                return Err(invalid("permutation must name each axis exactly once"));
            }
            seen[axis] = true;
        }
        let shape: Vec<_> = axes.iter().map(|&axis| self.shape[axis]).collect();
        let input_strides = strides(&self.shape);
        let mut tokens = allocate_tokens(self.len())?;
        for flat in 0..self.len() {
            let mut quotient = flat;
            let mut offset = 0;
            for axis in (0..shape.len()).rev() {
                let coordinate = quotient % shape[axis];
                quotient /= shape[axis];
                offset += coordinate * input_strides[axes[axis]];
            }
            tokens.push(self.tokens[offset]);
        }
        Ok(Self { tokens, shape })
    }

    pub fn slice(&self, axis: usize, start: usize, stop: usize, step: usize) -> Result<Self> {
        if axis >= self.ndim() {
            return Err(invalid("slice axis is outside the tensor"));
        }
        if step == 0 || start > stop || stop > self.shape[axis] {
            return Err(invalid("slice requires ordered bounds and positive step"));
        }
        let mut shape = self.shape.clone();
        shape[axis] = if start == stop {
            0
        } else {
            1 + (stop - start - 1) / step
        };
        let count = shape_len(&shape)?;
        let input_strides = strides(&self.shape);
        let mut tokens = allocate_tokens(count)?;
        for flat in 0..count {
            let mut quotient = flat;
            let mut offset = 0;
            for j in (0..shape.len()).rev() {
                let mut coordinate = quotient % shape[j];
                quotient /= shape[j];
                if j == axis {
                    coordinate = start + coordinate * step;
                }
                offset += coordinate * input_strides[j];
            }
            tokens.push(self.tokens[offset]);
        }
        Ok(Self { tokens, shape })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(phases: &[u32], shape: &[usize]) -> Tensor {
        Tensor::from_phases(phases.to_vec(), shape.to_vec()).unwrap()
    }

    #[test]
    fn exhaustive_domain_round_trip_and_literal_bytes() {
        assert_eq!(std::mem::size_of::<Token>(), 4);
        for phase in 0..MODULUS {
            let token = Token::from_phase(phase).unwrap();
            assert_eq!(token.phase(), phase);
        }
        assert_eq!(
            Token::from_phase(0).unwrap(),
            Token {
                r: 255,
                g: 0,
                b: 0,
                d: 0
            }
        );
        assert_eq!(
            Token::from_phase(255 * 256).unwrap(),
            Token {
                r: 255,
                g: 255,
                b: 0,
                d: 0
            }
        );
        assert_eq!(
            Token::from_phase(MODULUS - 1).unwrap(),
            Token {
                r: 255,
                g: 0,
                b: 1,
                d: 255
            }
        );
    }

    #[test]
    fn rgb_encoder_canonical_hues_and_achromatic_rejection() {
        for hue in 0..HUE_STEPS {
            let token = Token::from_phase(hue * 256).unwrap();
            assert_eq!(Token::from_rgb(token.r, token.g, token.b).unwrap(), token);
        }
        assert_eq!(Token::from_rgb(100, 50, 0).unwrap().phase(), 255 * 128);
        assert!(Token::from_rgb(0, 0, 0).is_err());
        assert!(Token::from_rgb(255, 255, 255).is_err());
        assert!(Token::new(254, 0, 0, 0).is_err());
        assert!(Token::new(255, 255, 255, 0).is_err());
    }

    #[test]
    fn ring_laws_and_all_byte_depth() {
        for seed in 0..1000u32 {
            let a = tensor(&[(seed * 997) % MODULUS], &[]);
            let b = tensor(&[(seed * 7919 + 7) % MODULUS], &[]);
            let c = tensor(&[(seed * 47 + 101) % MODULUS], &[]);
            assert_eq!(
                a.add(&b).unwrap().add(&c).unwrap(),
                a.add(&b.add(&c).unwrap()).unwrap()
            );
            assert_eq!(
                a.product(&b.add(&c).unwrap()).unwrap(),
                a.product(&b).unwrap().add(&a.product(&c).unwrap()).unwrap()
            );
            assert_eq!(a.add(&b).unwrap().sub(&b).unwrap(), a);
        }
        assert_eq!(
            tensor(&[255], &[])
                .add(&tensor(&[1], &[]))
                .unwrap()
                .tokens()[0],
            Token {
                r: 255,
                g: 1,
                b: 0,
                d: 0
            }
        );
        assert_eq!(
            tensor(&[MODULUS - 1], &[])
                .add(&tensor(&[1], &[]))
                .unwrap()
                .phases(),
            vec![0]
        );
        assert_eq!(
            tensor(&[MODULUS - 1], &[])
                .product(&tensor(&[MODULUS - 1], &[]))
                .unwrap()
                .phases(),
            vec![1]
        );
    }

    #[test]
    fn broadcasting_scalar_empty_and_rejection() {
        let a = tensor(&[1, 2], &[2, 1]);
        let b = tensor(&[10, 20, 30], &[1, 3]);
        assert_eq!(a.add(&b).unwrap().phases(), vec![11, 21, 31, 12, 22, 32]);
        assert_eq!(a.product(&tensor(&[2], &[])).unwrap().phases(), vec![2, 4]);
        let empty = tensor(&[], &[0, 3]);
        assert_eq!(empty.add(&tensor(&[2], &[])).unwrap().shape(), &[0, 3]);
        assert!(tensor(&[1, 2], &[2])
            .add(&tensor(&[1, 2, 3], &[3]))
            .is_err());
    }

    #[test]
    fn matrix_dot_sum_and_empty_contractions() {
        let a = tensor(&[1, 2, 3, 4, 5, 6], &[2, 3]);
        let b = tensor(&[7, 8, 9, 10, 11, 12], &[3, 2]);
        assert_eq!(a.matmul(&b).unwrap().phases(), vec![58, 64, 139, 154]);
        assert_eq!(a.sum(0).unwrap().phases(), vec![5, 7, 9]);
        assert_eq!(a.sum(1).unwrap().phases(), vec![6, 15]);
        assert_eq!(a.dot(&a).unwrap().phases(), vec![91]);
        assert_eq!(
            tensor(&[], &[2, 0])
                .matmul(&tensor(&[], &[0, 3]))
                .unwrap()
                .phases(),
            vec![0; 6]
        );
        assert_eq!(tensor(&[], &[2, 0]).sum(1).unwrap().phases(), vec![0; 2]);
        assert!(a.matmul(&a).is_err());
        assert!(a.dot(&b).is_err());
        assert!(a.sum(2).is_err());
    }

    #[test]
    fn native_shape_operations_preserve_bytes() {
        let a = tensor(&[0, 1, 2, 3, 4, 5], &[2, 3]);
        assert_eq!(a.permute(&[1, 0]).unwrap().phases(), vec![0, 3, 1, 4, 2, 5]);
        assert_eq!(a.permute(&[1, 0]).unwrap().permute(&[1, 0]).unwrap(), a);
        assert_eq!(a.slice(1, 0, 3, 2).unwrap().phases(), vec![0, 2, 3, 5]);
        assert_eq!(a.reshape(vec![3, 2]).unwrap().tokens(), a.tokens());
        assert_eq!(tensor(&[1], &[]).permute(&[]).unwrap().shape(), &[]);
        assert!(a.permute(&[1, 1]).is_err());
        assert!(a.slice(0, 0, 2, 0).is_err());
        assert!(a.reshape(vec![5]).is_err());
    }

    #[test]
    fn shortest_arc_blend_rotation_and_half_circle_tie() {
        let a = tensor(&[MODULUS - 10], &[]);
        let b = tensor(&[10], &[]);
        assert_eq!(a.blend(&b, 0).unwrap(), a);
        assert_eq!(a.blend(&b, 255).unwrap(), b);
        assert_eq!(a.blend(&b, 128).unwrap().phases(), vec![0]);
        assert_eq!(b.blend(&a, 128).unwrap().phases(), vec![0]);
        assert_eq!(
            tensor(&[0], &[])
                .blend(&tensor(&[MODULUS / 2], &[]), 255)
                .unwrap()
                .phases(),
            vec![MODULUS / 2]
        );
        assert_eq!(
            tensor(&[0], &[])
                .blend(&tensor(&[MODULUS / 2], &[]), 128)
                .unwrap()
                .phases(),
            vec![98304]
        );
        assert_eq!(
            tensor(&[MODULUS / 2], &[])
                .blend(&tensor(&[0], &[]), 128)
                .unwrap()
                .phases(),
            vec![294144]
        );
        assert_eq!(
            a.rotate(i64::MIN)
                .unwrap()
                .rotate(i64::MAX)
                .unwrap()
                .rotate(1)
                .unwrap(),
            a
        );
    }

    #[test]
    fn strict_dimensions_and_token_validation() {
        assert!(Tensor::from_phases(vec![0], vec![2]).is_err());
        assert!(Tensor::from_phases(vec![], vec![usize::MAX, 2]).is_err());
        assert!(Tensor::from_phases(vec![], vec![1; MAX_RANK + 1]).is_err());
        assert!(Tensor::from_phases(vec![MODULUS], vec![]).is_err());
        assert!(Tensor::from_rgbd(
            vec![Token {
                r: 0,
                g: 0,
                b: 0,
                d: 0
            }],
            vec![]
        )
        .is_err());
        let empty = Tensor::from_phases(vec![], vec![0; MAX_RANK]).unwrap();
        assert_eq!(empty.add(&tensor(&[1], &[])).unwrap(), empty);
    }
}
