// Import necessary types.
use crypto_bigint::{U128, U192, U384, Encoding};

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::marker::PhantomData;

// Define 5 distinct large 64-bit primes.
pub const P64_1: u64 = 18446744073709551557; // 2^64 - 59
pub const P64_2: u64 = 18446744073709551533; // 2^64 - 83
pub const P64_3: u64 = 18446744073709551521; // 2^64 - 95
pub const P64_4: u64 = 18446744073709551437; // 2^64 - 179
pub const P64_5: u64 = 18446744073709551427; // 2^64 - 189

// Primes for Task 1 and 1B (5 primes, ~320 bits)
pub const PRIMES_TASK1: [u64; 5] = [P64_1, P64_2, P64_3, P64_4, P64_5];
// Primes for Task 2 (3 primes, ~192 bits)
pub const PRIMES_TASK2: [u64; 3] = [P64_1, P64_2, P64_3];


// --- CRT Context Implementation (Garner's Algorithm) ---

// Helper function to compute modular inverse using Extended Euclidean Algorithm.
// Returns (a^-1) mod m. Uses i128 for intermediate calculations.
fn mod_inverse_u64(a: u64, m: u64) -> u64 {
    let a = a as i128;
    let m = m as i128;
    let mut t = 0i128;
    let mut new_t = 1i128;
    let mut r = m;
    let mut new_r = a;

    while new_r != 0 {
        // Quotient calculation handles potential negative new_r if inputs weren't primes, though here they are.
        let quotient = r.checked_div(new_r).expect("Division by zero");
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if r > 1 {
        panic!("a={} is not invertible mod m={}", a, m);
    }
    if t < 0 {
        t += m;
    }

    t as u64
}


// Define a helper trait to abstract the requirements for the CRT result type.
// We use specific wrapping methods because std::ops::Mul performs widening multiplication in v0.5.5.
pub trait CrtInteger: Copy + From<u64> + Send + Sync
{
    const ONE: Self;
    const ZERO: Self;
    // Methods that guarantee the output is Self
    fn wrapping_mul(&self, rhs: &Self) -> Self;
    fn wrapping_add(&self, rhs: &Self) -> Self;
}

// Implement the trait for the required types.
impl CrtInteger for U192 {
    const ONE: Self = U192::ONE;
    const ZERO: Self = U192::ZERO;
    fn wrapping_mul(&self, rhs: &Self) -> Self { self.wrapping_mul(rhs) }
    fn wrapping_add(&self, rhs: &Self) -> Self { self.wrapping_add(rhs) }
}
impl CrtInteger for U384 {
    const ONE: Self = U384::ONE;
    const ZERO: Self = U384::ZERO;
    fn wrapping_mul(&self, rhs: &Self) -> Self { self.wrapping_mul(rhs) }
    fn wrapping_add(&self, rhs: &Self) -> Self { self.wrapping_add(rhs) }
}


// CrtContext holds precomputed constants for CRT reconstruction.
#[derive(Clone)]
pub struct CrtContext<T: CrtInteger, const K: usize> {
    // Precomputed (P_j)^{-1} mod P_i for j < i.
    inverses: Vec<u64>,
    primes: [u64; K],
    // Primes as T (for the final reconstruction step)
    primes_t: Vec<T>,
    _marker: PhantomData<T>,
}

impl<T: CrtInteger, const K: usize> CrtContext<T, K>
{
    pub fn new(primes: [u64; K]) -> Self {
        let mut inverses = Vec::with_capacity(K * (K - 1) / 2);
        let mut primes_t = Vec::with_capacity(K);

        for &p in primes.iter() {
            primes_t.push(T::from(p));
        }

        // Precompute inverses for Garner's algorithm.
        for i in 1..K {
            let p_i = primes[i];
            
            for j in 0..i {
                let p_j = primes[j];

                // Calculate inv = (p_j)^{-1} mod p_i using the local helper function.
                let inv = mod_inverse_u64(p_j, p_i);
                
                inverses.push(inv);
            }
        }

        Self {
            inverses,
            primes,
            primes_t,
            _marker: PhantomData,
        }
    }

    // Reconstruct the value using Garner's algorithm (Mixed-Radix Representation).
    pub fn reconstruct(&self, residues: &[u64; K]) -> T {
        // 1. Calculate the mixed-radix coefficients (v_i)
        let mut v = [0u64; K];
        v[0] = residues[0];

        let mut inv_idx = 0;
        for i in 1..K {
            let mut temp = residues[i];
            let p_i = self.primes[i];

            for j in 0..i {
                // temp = (temp - v_j) * inv(p_j, p_i) mod p_i

                // Robust modular subtraction. 
                let v_j_mod_pi = v[j] % p_i;

                let diff = if temp >= v_j_mod_pi {
                    temp - v_j_mod_pi
                } else {
                    p_i - (v_j_mod_pi - temp)
                };

                let inv = self.inverses[inv_idx];
                inv_idx += 1;
                
                // Use u128 for intermediate product
                temp = ((diff as u128 * inv as u128) % p_i as u128) as u64;
            }
            v[i] = temp;
        }

        // 2. Reconstruct the final value x = v_0 + v_1*p_0 + v_2*p_0*p_1 + ...
        let mut result = T::from(v[0]);
        let mut coefficient = T::ONE;

        // Use the specific wrapping methods defined in the CrtInteger trait.
        for i in 1..K {
            // We know the coefficient fits in T (by CRT design), so wrapping_mul is safe.
            coefficient = coefficient.wrapping_mul(&self.primes_t[i-1]);
            
            let v_i_t = T::from(v[i]);
            // We know the term fits in T, so wrapping_mul is safe.
            let term = coefficient.wrapping_mul(&v_i_t);
            // wrapping_add is safe.
            result = result.wrapping_add(&term);
        }

        result
    }
}

// Specific contexts for the tasks
pub type CrtContextTask1 = CrtContext<U384, 5>;
pub type CrtContextTask2 = CrtContext<U192, 3>;

// Initialize the global contexts
pub fn initialize_crt_contexts() -> (CrtContextTask1, CrtContextTask2) {
    (CrtContextTask1::new(PRIMES_TASK1), CrtContextTask2::new(PRIMES_TASK2))
}


// --- Existing Helper Functions ---

// A 128-bit prime for Tasks 3 and 4 (2^128 - 159)
pub fn get_128bit_prime() -> U128 {
    U128::MAX.saturating_sub(&U128::from_u8(158))
}

pub fn random_u128<R: Rng>(rng: &mut R) -> U128 {
    U128::from_le_bytes(rng.gen())
}

// Helper to convert crypto_bigint::U128 to primitive u128 for CRT tasks.
pub fn to_primitive(x: &U128) -> u128 {
    (*x).try_into().unwrap()
}

// Data structures
pub type BitVector = Vec<u64>;
pub type SmallIntVector = Vec<u8>; // For Task 1B

// Generates bit vectors.
pub fn generate_bit_vectors(count: usize, n: usize, rng: &mut impl Rng) -> Vec<BitVector> {
    assert!(n % 64 == 0, "n must be a multiple of 64 for this implementation");
    let n_chunks = n / 64;
    (0..count).map(|_| {
        (0..n_chunks).map(|_| rng.gen::<u64>()).collect()
    }).collect()
}

// Generates vectors for Task 1B.
pub fn generate_small_int_vectors(count: usize, n: usize, rng: &mut impl Rng) -> Vec<SmallIntVector> {
    assert!(n.is_power_of_two(), "n must be a power of two");
    let log_n = n.ilog2();
    let max_val = 1 + log_n as u8;

    if max_val > 15 {
        panic!("N={} is too large for the 4-bit M4R implementation.", n);
    }

    (0..count).map(|_| {
        (0..n).map(|_| rng.gen_range(0..=max_val)).collect()
    }).collect()
}


// Transpose the bit vectors for optimized M4R access (Tasks 1 and 3).
pub fn transpose_bit_vectors(vectors: &[BitVector], m: usize, n: usize) -> Vec<u32> {
    let num_vectors = 32 * m;
    assert_eq!(vectors.len(), num_vectors);

    let mut transposed = vec![0u32; n * m];

    transposed.par_chunks_mut(m).enumerate().for_each(|(k, chunk)| {
        let chunk_idx = k / 64;
        let pos_in_chunk = k % 64;

        for i in 0..m {
            let mut word = 0u32;
            for j in 0..32 {
                let vector_idx = i * 32 + j;
                let data_chunk = vectors[vector_idx][chunk_idx];
                if (data_chunk >> pos_in_chunk) & 1 == 1 {
                    word |= 1 << j;
                }
            }
            chunk[i] = word;
        }
    });

    transposed
}

// Transpose and pack vectors for Task 1B.
pub fn transpose_and_pack_small_int_vectors(vectors: &[SmallIntVector], m: usize, n: usize) -> Vec<u128> {
    let num_vectors = 32 * m;
    assert_eq!(vectors.len(), num_vectors);

    let mut transposed = vec![0u128; n * m];

    transposed.par_chunks_mut(m).enumerate().for_each(|(k, chunk)| {
        for i in 0..m {
            let mut word = 0u128;
            for j in 0..32 {
                let vector_idx = i * 32 + j;
                let entry = vectors[vector_idx][k] as u128;
                word |= entry << (j * 4);
            }
            chunk[i] = word;
        }
    });

    transposed
}


pub fn generate_u32_vectors(m: usize, n: usize, rng: &mut impl Rng) -> Vec<Vec<u32>> {
    (0..m).map(|_| {
        (0..n).map(|_| rng.gen::<u32>()).collect()
    }).collect()
}

// Generate standardized data for benchmarking
pub fn generate_benchmark_data(m: usize, n: usize) -> (
    [U128; 32], Vec<U128>, Vec<U128>, Vec<BitVector>, Vec<SmallIntVector>, Vec<Vec<u32>>
) {
    assert!(m.is_power_of_two(), "m must be a power of two");
    let t = m.ilog2() as usize;

    if n % 64 != 0 {
        panic!("N={} must be a multiple of 64.", n);
    }
    
    let mut rng = StdRng::seed_from_u64(42);

    let a = [0; 32].map(|_| random_u128(&mut rng));
    let b = (0..m).map(|_| random_u128(&mut rng)).collect();
    let b_t = (0..t).map(|_| random_u128(&mut rng)).collect();

    let u_bits = generate_bit_vectors(32 * m, n, &mut rng);
    let u_small_int = generate_small_int_vectors(32 * m, n, &mut rng);
    let u_u32 = generate_u32_vectors(m, n, &mut rng);

    (a, b, b_t, u_bits, u_small_int, u_u32)
}