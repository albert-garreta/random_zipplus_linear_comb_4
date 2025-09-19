use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
// Import task1b
use vector_benchmarks::{utils::*, task1, task1b, task2, task3, task4};
// Import U128 and DynResidueParams.
use crypto_bigint::{U128, modular::runtime_mod::DynResidueParams};
use std::sync::Arc;

// Define benchmark parameters (M power of 2, N multiple of 64 and power of 2)
const M_VALUES: [usize; 1] = [1<<7]; // 256, 1024, 4096
const N_VALUES: [usize; 1] = [1<<13]; 

// Define LIMBS constant using U128::LIMBS.
const LIMBS: usize = U128::LIMBS;

fn bench_tasks(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorTasks");

    for &m in &M_VALUES {
        for &n in &N_VALUES {
            
            let input_size = format!("m={}, n={}", m, n);

            // Throughput estimation based on O(m*n) operations
            group.throughput(Throughput::Elements((m as u64 * n as u64)));

            // --- Data Generation (Excluded from benchmark) ---
            let (a, b, b_t, u_bits, u_small_int, u_u32) = generate_benchmark_data(m, n);
            
            // Transposition (Excluded from benchmark)
            let u_transposed_bits = transpose_bit_vectors(&u_bits, m, n);
            let u_packed_small_int = transpose_and_pack_small_int_vectors(&u_small_int, m, n);


            // --- Task 1 Setup (Original - Bits) ---
            let tables1 = task1::preprocess_m4r(&a);

            group.bench_with_input(BenchmarkId::new("Task1 (Bits+CRT+M4R_8bit)", &input_size), &(m, n), |bencher, &(_m, _n)| {
                bencher.iter(|| {
                    task1::task1(&b, &u_transposed_bits, m, n, &tables1)
                });
            });

            // --- Task 1B Setup (New - Small Ints) ---
            // M4R Preprocessing (Excluded from benchmark)
            let tables1b = task1b::preprocess_m4r(&a);

            group.bench_with_input(BenchmarkId::new("Task1B (SmallInt+CRT+M4R_16bit)", &input_size), &(m, n), |bencher, &(_m, _n)| {
                bencher.iter(|| {
                    task1b::task1b(&b, &u_packed_small_int, m, n, &tables1b)
                });
            });


            // --- Task 2 Setup ---
            group.bench_with_input(BenchmarkId::new("Task2 (U32+CRT)", &input_size), &(m, n), |bencher, &(_m, _n)| {
                bencher.iter(|| {
                    task2::task2(&b, &u_u32, m, n)
                });
            });

            // --- Task 3 Setup ---
            let q = get_128bit_prime();
            
            // M4R Preprocessing (Excluded from benchmark)
            let monty_params_setup: Arc<DynResidueParams<LIMBS>> = Arc::new(DynResidueParams::new(&q));
            let tables3 = task3::preprocess_m4r_montgomery(&a, monty_params_setup.clone());

            group.bench_with_input(BenchmarkId::new("Task3 (Bits+Monty+M4R)", &input_size), &(m, n), |bencher, &(_m, _n)| {
                bencher.iter(|| {
                    // Benchmark includes Montgomery setup and mu_i generation.
                    task3::task3(q, &b_t, &u_transposed_bits, m, n, &tables3)
                });
            });

            // --- Task 4 Setup ---
            group.bench_with_input(BenchmarkId::new("Task4 (U32+Monty)", &input_size), &(m, n), |bencher, &(_m, _n)| {
                bencher.iter(|| {
                    // Benchmark includes Montgomery setup and mu_i generation.
                    task4::task4(q, &b_t, &u_u32, m, n)
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_tasks);
criterion_main!(benches);