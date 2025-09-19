// Import Encoding trait to use from_le_bytes.
use crypto_bigint::{U128, Encoding};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;

// A 64-bit prime for the CRT approach (2^64 - 59)
pub const P64: u64 = 18446744073709551557;

// A 128-bit prime for Tasks 3 and 4 (2^128 - 159)
pub fn get_128bit_prime() -> U128 {
    // saturating_sub requires a reference (&).
    U128::MAX.saturating_sub(&U128::from_u8(158))
}

pub fn random_u128<R: Rng>(rng: &mut R) -> U128 {
    U128::from_le_bytes(rng.gen())
}

// Helper to convert crypto_bigint::U128 to primitive u128 for CRT tasks.
pub fn to_primitive(x: &U128) -> u128 {
    // U128 implements Copy. Dereference x to use the infallible TryFrom<U128> implementation.
    (*x).try_into().unwrap()
}

// Data structures
pub type BitVector = Vec<u64>;
pub type SmallIntVector = Vec<u8>; // For Task 1B

// Generates bit vectors. Assumes n is a multiple of 64.
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
    // The range is [0, 1 + log2(n)] inclusive.
    let max_val = 1 + log_n as u8;

    // Check constraint for M4R implementation (max 4 bits). 
    // Range must be [0, 15], so max_val must be <= 15.
    if max_val > 15 {
        // If max_val = 16 (N=32768), we need 5 bits.
        panic!("N={} is too large for the 4-bit M4R implementation. Max supported N is 16384 (logN=14, max_val=15).", n);
    }

    (0..count).map(|_| {
        // Generate entries in the range [0, max_val]
        (0..n).map(|_| rng.gen_range(0..=max_val)).collect()
    }).collect()
}


// Transpose the bit vectors for optimized M4R access (Tasks 1 and 3).
pub fn transpose_bit_vectors(vectors: &[BitVector], m: usize, n: usize) -> Vec<u32> {
    let num_vectors = 32 * m;
    assert_eq!(vectors.len(), num_vectors);

    let mut transposed = vec![0u32; n * m];

    // Parallelize the transposition using Rayon. Iterate over positions k (0..n).
    transposed.par_chunks_mut(m).enumerate().for_each(|(k, chunk)| {
        let chunk_idx = k / 64;
        let pos_in_chunk = k % 64;

        for i in 0..m {
            // Fill chunk[i] (block i at position k)
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
// Input: 32*m vectors of length n (entries are 4-bit).
// Output: Flattened Vec<u128> of size n*m.
// transposed[k*m + i] holds the 128 bits (32 entries * 4 bits) corresponding to block i at position k.
pub fn transpose_and_pack_small_int_vectors(vectors: &[SmallIntVector], m: usize, n: usize) -> Vec<u128> {
    let num_vectors = 32 * m;
    assert_eq!(vectors.len(), num_vectors);

    let mut transposed = vec![0u128; n * m];

    // Parallelize the transposition using Rayon. Iterate over positions k (0..n).
    transposed.par_chunks_mut(m).enumerate().for_each(|(k, chunk)| {
        for i in 0..m {
            // Fill chunk[i] (block i at position k)
            let mut word = 0u128;
            for j in 0..32 {
                let vector_idx = i * 32 + j;
                // Accessing vectors[vector_idx][k] (column access on row-major data).
                let entry = vectors[vector_idx][k] as u128;
                
                // Pack the 4-bit entry into the 128-bit word.
                // j=0 is the least significant 4 bits.
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