//! numpy's `default_rng`: `SeedSequence` → `PCG64` → `Generator`, draw by draw.
//!
//! Mirrors numpy 2.5.3's `numpy/random/bit_generator.pyx` (SeedSequence),
//! `numpy/random/src/pcg64/pcg64.h` (the XSL-RR 128/64 generator and its
//! buffered 32-bit output), `numpy/random/src/distributions/distributions.c`
//! (the draws) and `numpy/random/_generator.pyx` / `_bounded_integers.pyx.in`
//! (which draw each `Generator` method makes, in which order).
//!
//! Only the paths Virtual Impressionist and yapocis_external use are ported:
//! `int64` integers without `use_masked`, float64 and float32 `random`, the
//! ziggurat normal and exponential, `geometric`, `uniform`, `normal`,
//! `exponential`, `choice` without replacement and without `p`, `permutation`
//! and `shuffle` of 1-D data.

mod ziggurat;

use ziggurat::*;

// --- SeedSequence ------------------------------------------------------------

const DEFAULT_POOL_SIZE: usize = 4;
const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

/// `np.random.SeedSequence(entropy)` for integer or integer-sequence entropy,
/// with no spawn key.
#[derive(Debug, Clone)]
pub struct SeedSequence {
    pool: [u32; DEFAULT_POOL_SIZE],
}

/// `_int_to_uint32_array(n)`: little-endian 32-bit words of a non-negative int
/// (a single 0 word for 0).
fn int_to_u32_words(mut n: u128, out: &mut Vec<u32>) {
    if n == 0 {
        out.push(0);
    }
    while n > 0 {
        out.push((n & 0xFFFF_FFFF) as u32);
        n >>= 32;
    }
}

fn hashmix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^= value >> XSHIFT;
    value
}

fn mix(x: u32, y: u32) -> u32 {
    let mut result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

impl SeedSequence {
    /// `SeedSequence(entropy)` where `entropy` is a non-negative int, or a list
    /// of them: each becomes its uint32 words and the words are concatenated,
    /// as `_coerce_to_uint32_array` does.
    pub fn new(entropy: &[u128]) -> SeedSequence {
        let mut words = Vec::new();
        for &e in entropy {
            int_to_u32_words(e, &mut words);
        }
        Self::from_words(&words)
    }

    /// `SeedSequence` from an already-coerced uint32 entropy array.
    pub fn from_words(entropy: &[u32]) -> SeedSequence {
        let mut mixer = [0u32; DEFAULT_POOL_SIZE];
        let mut hash_const = INIT_A;
        for (i, m) in mixer.iter_mut().enumerate() {
            *m = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        for i_src in 0..DEFAULT_POOL_SIZE {
            for i_dst in 0..DEFAULT_POOL_SIZE {
                if i_src != i_dst {
                    mixer[i_dst] = mix(mixer[i_dst], hashmix(mixer[i_src], &mut hash_const));
                }
            }
        }
        for &e in entropy.iter().skip(DEFAULT_POOL_SIZE) {
            for m in mixer.iter_mut() {
                *m = mix(*m, hashmix(e, &mut hash_const));
            }
        }
        SeedSequence { pool: mixer }
    }

    /// `generate_state(n_words, np.uint32)`.
    pub fn generate_state_u32(&self, n_words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        let mut state = Vec::with_capacity(n_words);
        for i in 0..n_words {
            let mut data_val = self.pool[i % DEFAULT_POOL_SIZE];
            data_val ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data_val = data_val.wrapping_mul(hash_const);
            data_val ^= data_val >> XSHIFT;
            state.push(data_val);
        }
        state
    }

    /// `generate_state(n_words, np.uint64)`: twice the 32-bit words, viewed as
    /// little-endian uint64 pairs.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        self.generate_state_u32(n_words * 2)
            .chunks(2)
            .map(|p| (p[0] as u64) | ((p[1] as u64) << 32))
            .collect()
    }
}

// --- PCG64 (XSL-RR 128/64, "setseq") -------------------------------------------

const PCG_DEFAULT_MULTIPLIER_128: u128 =
    ((2_549_297_995_355_413_924u128) << 64) + 4_865_540_595_714_422_341u128;

/// numpy's `PCG64` bit generator, with its buffered 32-bit output.
#[derive(Debug, Clone)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    /// `PCG64(seed_seq)`: `generate_state(4, uint64)` gives the 128-bit state
    /// (words 0, 1: high, low) and the stream (words 2, 3), then
    /// `pcg_setseq_128_srandom_r`.
    pub fn from_seed_sequence(seq: &SeedSequence) -> Pcg64 {
        let w = seq.generate_state_u64(4);
        let initstate = ((w[0] as u128) << 64) | w[1] as u128;
        let initseq = ((w[2] as u128) << 64) | w[3] as u128;
        let mut rng = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    #[inline]
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_DEFAULT_MULTIPLIER_128)
            .wrapping_add(self.inc);
    }

    #[inline]
    fn output(state: u128) -> u64 {
        let high = (state >> 64) as u64;
        let low = state as u64;
        (high ^ low).rotate_right((high >> 58) as u32)
    }

    /// `pcg64_next64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        Self::output(self.state)
    }

    /// `pcg64_next32`: the low half of a 64-bit draw, then its buffered high half.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        next as u32
    }

    /// `next_double`: `(next_uint64 >> 11) * (1 / 2^53)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `next_float`: `(next_uint32 >> 8) * (1 / 2^24)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0f32 / 16_777_216.0f32)
    }
}

// --- Generator ---------------------------------------------------------------------

/// Whether numpy's C was compiled with `a + b * c` contracted into a fused
/// multiply-add. clang contracts on arm64 (the macOS wheels); the manylinux
/// x86-64 wheels do not. It changes the last bit of `normal`, `uniform` and the
/// ziggurat tail tests, so a port must use the host's policy; the host fixtures
/// record it (`index.json` `"fma"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contraction {
    /// `a + b * c` as two roundings.
    Plain,
    /// `fma(b, c, a)`, one rounding.
    Fma,
}

impl Contraction {
    /// The policy of this build target's numpy wheels, absent other evidence.
    pub fn host_default() -> Contraction {
        if cfg!(target_arch = "aarch64") {
            Contraction::Fma
        } else {
            Contraction::Plain
        }
    }

    #[inline]
    /// `a + b * c`, fused or not by this policy.
    pub fn mul_add(self, b: f64, c: f64, a: f64) -> f64 {
        match self {
            Contraction::Plain => a + b * c,
            Contraction::Fma => b.mul_add(c, a),
        }
    }
}

/// `np.random.Generator(PCG64(seed_seq))`, i.e. `np.random.default_rng(seed)`.
#[derive(Debug, Clone)]
pub struct Generator {
    bitgen: Pcg64,
    contraction: Contraction,
}

impl Generator {
    /// `np.random.default_rng(seed)` for an int or a list of ints.
    pub fn default_rng(entropy: &[u128]) -> Generator {
        Generator::from_seed_sequence(&SeedSequence::new(entropy))
    }

    pub fn from_seed_sequence(seq: &SeedSequence) -> Generator {
        Generator {
            bitgen: Pcg64::from_seed_sequence(seq),
            contraction: Contraction::host_default(),
        }
    }

    /// Use the given FMA contraction policy instead of the host default.
    pub fn with_contraction(mut self, contraction: Contraction) -> Generator {
        self.contraction = contraction;
        self
    }

    pub fn contraction(&self) -> Contraction {
        self.contraction
    }

    pub fn bit_generator(&mut self) -> &mut Pcg64 {
        &mut self.bitgen
    }

    // -- uniform floats ---------------------------------------------------------

    /// `rng.random()` / `rng.random(size)` as float64, in C order.
    pub fn random_f64(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.bitgen.next_f64()).collect()
    }

    /// `rng.random(size, dtype=np.float32)`.
    pub fn random_f32(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.bitgen.next_f32()).collect()
    }

    /// `rng.uniform(low, high, size)` with scalar bounds:
    /// `low + (high - low) * next_double`.
    pub fn uniform(&mut self, low: f64, high: f64, n: usize) -> Vec<f64> {
        let range = high - low;
        let c = self.contraction;
        (0..n)
            .map(|_| c.mul_add(range, self.bitgen.next_f64(), low))
            .collect()
    }

    // -- bounded integers (`integers`, dtype int64, use_masked False) --------------

    /// `buffered_bounded_lemire_uint32` (the buffer arguments are unused in
    /// numpy: it draws through the bit generator's own 32-bit buffer).
    fn bounded_lemire_u32(&mut self, rng: u32) -> u32 {
        let rng_excl = rng.wrapping_add(1);
        debug_assert!(rng != u32::MAX);
        let mut m = (self.bitgen.next_u32() as u64) * (rng_excl as u64);
        let mut leftover = (m & 0xFFFF_FFFF) as u32;
        if leftover < rng_excl {
            let threshold = (u32::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = (self.bitgen.next_u32() as u64) * (rng_excl as u64);
                leftover = (m & 0xFFFF_FFFF) as u32;
            }
        }
        (m >> 32) as u32
    }

    /// `bounded_lemire_uint64`.
    fn bounded_lemire_u64(&mut self, rng: u64) -> u64 {
        let rng_excl = rng.wrapping_add(1);
        debug_assert!(rng != u64::MAX);
        let mut m = (self.bitgen.next_u64() as u128) * (rng_excl as u128);
        let mut leftover = m as u64;
        if leftover < rng_excl {
            let threshold = (u64::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = (self.bitgen.next_u64() as u128) * (rng_excl as u128);
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// `random_bounded_uint64(state, off, rng, mask, use_masked=False)`: one value
    /// in `off ..= off + rng`. The same draws as one step of
    /// `random_bounded_uint64_fill`.
    pub fn bounded_u64(&mut self, off: u64, rng: u64) -> u64 {
        if rng == 0 {
            off
        } else if rng <= 0xFFFF_FFFF {
            if rng == 0xFFFF_FFFF {
                off.wrapping_add(self.bitgen.next_u32() as u64)
            } else {
                off.wrapping_add(self.bounded_lemire_u32(rng as u32) as u64)
            }
        } else if rng == u64::MAX {
            off.wrapping_add(self.bitgen.next_u64())
        } else {
            off.wrapping_add(self.bounded_lemire_u64(rng))
        }
    }

    /// `rng.integers(low, high, size=n)` with scalar int bounds, dtype int64,
    /// `endpoint=False`: `n` values in `low .. high`.
    pub fn integers(&mut self, low: i64, high: i64, n: usize) -> Vec<i64> {
        assert!(low < high, "low >= high");
        let rng = (high - 1).wrapping_sub(low) as u64;
        let off = low as u64;
        (0..n).map(|_| self.bounded_u64(off, rng) as i64).collect()
    }

    /// `rng.integers(high)` for one scalar: the same as `integers(0, high, 1)[0]`.
    pub fn integer(&mut self, low: i64, high: i64) -> i64 {
        self.integers(low, high, 1)[0]
    }

    /// `rng.integers(low, highs)` with an array of exclusive upper bounds
    /// (`_rand_int64_broadcast`): one value per bound, in order.
    pub fn integers_broadcast(&mut self, low: i64, highs: &[i64]) -> Vec<i64> {
        highs
            .iter()
            .map(|&high| {
                assert!(low < high, "low >= high");
                let rng = (high - 1).wrapping_sub(low) as u64;
                self.bounded_u64(low as u64, rng) as i64
            })
            .collect()
    }

    /// `random_interval(max)`: masked rejection in `0 ..= max`, what `shuffle` and
    /// `permutation` draw.
    pub fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mask = gen_mask(max);
        if max <= 0xFFFF_FFFF {
            loop {
                let value = (self.bitgen.next_u32() as u64) & mask;
                if value <= max {
                    return value;
                }
            }
        } else {
            loop {
                let value = self.bitgen.next_u64() & mask;
                if value <= max {
                    return value;
                }
            }
        }
    }

    // -- distributions ------------------------------------------------------------

    /// `random_standard_normal` (ziggurat, float64).
    pub fn standard_normal_one(&mut self) -> f64 {
        loop {
            let mut r = self.bitgen.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * WI_DOUBLE[idx];
            if sign & 0x1 == 1 {
                x = -x;
            }
            if rabs < KI_DOUBLE[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.bitgen.next_f64()).ln_1p();
                    let yy = -(-self.bitgen.next_f64()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 == 1 {
                            -(ZIGGURAT_NOR_R + xx)
                        } else {
                            ZIGGURAT_NOR_R + xx
                        };
                    }
                }
            } else if self.contraction.mul_add(
                FI_DOUBLE[idx - 1] - FI_DOUBLE[idx],
                self.bitgen.next_f64(),
                FI_DOUBLE[idx],
            ) < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }

    /// `rng.standard_normal(size)` (float64).
    pub fn standard_normal(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.standard_normal_one()).collect()
    }

    /// `rng.normal(loc, scale, size)` with scalar parameters.
    pub fn normal(&mut self, loc: f64, scale: f64, n: usize) -> Vec<f64> {
        let c = self.contraction;
        (0..n)
            .map(|_| {
                let z = self.standard_normal_one();
                c.mul_add(scale, z, loc)
            })
            .collect()
    }

    /// `random_standard_exponential` (ziggurat, float64).
    pub fn standard_exponential_one(&mut self) -> f64 {
        let mut ri = self.bitgen.next_u64();
        ri >>= 3;
        let idx = (ri & 0xFF) as usize;
        ri >>= 8;
        let x = ri as f64 * WE_DOUBLE[idx];
        if ri < KE_DOUBLE[idx] {
            return x;
        }
        self.standard_exponential_unlikely(idx, x)
    }

    fn standard_exponential_unlikely(&mut self, idx: usize, x: f64) -> f64 {
        if idx == 0 {
            ZIGGURAT_EXP_R - (-self.bitgen.next_f64()).ln_1p()
        } else if self.contraction.mul_add(
            FE_DOUBLE[idx - 1] - FE_DOUBLE[idx],
            self.bitgen.next_f64(),
            FE_DOUBLE[idx],
        ) < (-x).exp()
        {
            x
        } else {
            self.standard_exponential_one()
        }
    }

    /// `rng.standard_exponential(size)` (float64, method 'zig').
    pub fn standard_exponential(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.standard_exponential_one()).collect()
    }

    /// `rng.exponential(scale, size)` with a scalar scale.
    pub fn exponential(&mut self, scale: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| scale * self.standard_exponential_one())
            .collect()
    }

    /// `random_geometric(p)`: search for p >= 1/3, inversion below.
    pub fn geometric_one(&mut self, p: f64) -> i64 {
        // numpy writes 0.333333333333333333333333; it is the f64 nearest to 1/3
        #[allow(clippy::excessive_precision)]
        const THIRD: f64 = 0.333_333_333_333_333_333_333_333;
        if p >= THIRD {
            let mut x: i64 = 1;
            let mut sum = p;
            let mut prod = p;
            let q = 1.0 - p;
            let u = self.bitgen.next_f64();
            while u > sum {
                prod *= q;
                sum += prod;
                x += 1;
            }
            x
        } else {
            let z = (-self.standard_exponential_one() / (-p).ln_1p()).ceil();
            if z >= 9.223_372_036_854_776e18 {
                i64::MAX
            } else {
                z as i64
            }
        }
    }

    /// `rng.geometric(p, size)` with a scalar p.
    pub fn geometric(&mut self, p: f64, n: usize) -> Vec<i64> {
        (0..n).map(|_| self.geometric_one(p)).collect()
    }

    // -- permutations ---------------------------------------------------------------

    /// `rng.shuffle(x)` of a 1-D array: Fisher-Yates from the end with
    /// `random_interval`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.random_interval(i as u64) as usize;
            x.swap(i, j);
        }
    }

    /// `rng.permutation(n)`: `arange(n)` shuffled.
    pub fn permutation(&mut self, n: usize) -> Vec<i64> {
        let mut idx: Vec<i64> = (0..n as i64).collect();
        self.shuffle(&mut idx);
        idx
    }

    /// `_shuffle_int(n, first, data)`: Fisher-Yates over `data[first..n]` with
    /// `random_bounded_uint64` (Lemire), what `choice` uses.
    fn shuffle_int(&mut self, first: usize, data: &mut [i64]) {
        for i in (first..data.len()).rev() {
            let j = self.bounded_u64(0, i as u64) as usize;
            data.swap(i, j);
        }
    }

    /// `rng.choice(pop_size, size, replace=False)` without `p`: the chosen
    /// indices, `size` of them, in numpy's order (`shuffle=True`).
    pub fn choice_without_replacement(&mut self, pop_size: usize, size: usize) -> Vec<i64> {
        assert!(size <= pop_size, "sample larger than population");
        let cutoff = 50;
        if pop_size > 10_000 && size > pop_size / cutoff {
            let mut idx: Vec<i64> = (0..pop_size as i64).collect();
            self.shuffle_int((pop_size - size).max(1), &mut idx);
            idx[pop_size - size..].to_vec()
        } else {
            let mut idx = vec![0i64; size];
            let mut set_size = (1.2 * size as f64) as u64;
            let mask = gen_mask(set_size);
            set_size = 1 + mask;
            let mut hash_set = vec![u64::MAX; set_size as usize];
            for j in (pop_size - size)..pop_size {
                let val = self.bounded_u64(0, j as u64);
                let mut loc = (val & mask) as usize;
                while hash_set[loc] != u64::MAX && hash_set[loc] != val {
                    loc = ((loc as u64 + 1) & mask) as usize;
                }
                if hash_set[loc] == u64::MAX {
                    hash_set[loc] = val;
                    idx[j - pop_size + size] = val as i64;
                } else {
                    let mut loc = (j as u64 & mask) as usize;
                    while hash_set[loc] != u64::MAX {
                        loc = ((loc as u64 + 1) & mask) as usize;
                    }
                    hash_set[loc] = j as u64;
                    idx[j - pop_size + size] = j as i64;
                }
            }
            self.shuffle_int(1, &mut idx);
            idx
        }
    }
}

/// `gen_mask` / `_gen_mask`: the smallest all-ones mask covering `max`.
fn gen_mask(max: u64) -> u64 {
    let mut mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_sequence_words_match_numpy_docstring() {
        // _coerce_to_uint32_array([12345, 0x10deadbeef, 67890, 0xdeadbeef])
        let mut w = Vec::new();
        for e in [12345u128, 0x10deadbeef, 67890, 0xdeadbeef] {
            int_to_u32_words(e, &mut w);
        }
        assert_eq!(w, [12345, 3735928559, 16, 67890, 3735928559]);
    }

    #[test]
    fn default_rng_zero_first_draws() {
        // np.random.default_rng(0).random() == 0.6369616873214543 and
        // np.random.default_rng(0).integers(10, size=3) -> [8, 6, 5] (numpy docs)
        let mut g = Generator::default_rng(&[0]);
        assert_eq!(g.random_f64(1)[0], 0.636_961_687_321_454_3);
        let mut g = Generator::default_rng(&[0]);
        assert_eq!(g.integers(0, 10, 3), vec![8, 6, 5]);
    }
}
