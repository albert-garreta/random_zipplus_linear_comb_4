use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use vector_benchmarks::{utils::*, task1, task1b, task2, task3, task4};
use crypto_bigint::{U128, modular::runtime_mod::DynResidueParams};
use std::sync::Arc;

// Define benchmark parameters
const M_VALUES: [usize; 1] = [1<<3]; // 256, 1024, 4096
const N_VALUES: [usize; 1] = [1<<8]; 

const LIMBS_U128: usize = U128::LIMBS;

fn bench_tasks(c: &mut Criterion) {
    let mut group = c.benchmark_group("VectorTasks");

    // Initialize CRT Contexts (Excluded from benchmark)
    let (crt_context1, crt_context2) = initialize_crt_contexts();

    for &m in &M_VALUES {
        for &n in &N_VALUES {
            
            let input_size = format!("m={}, n={}", m, n);
            // Throughput is O(m*n). The measured time includes the overhead of K primes and reconstruction.
            group.throughput(Throughput::Elements((m as u64 * n as u64)));

            // --- Data Generation (Excluded from benchmark) ---
            let (a, b, b_t, u_bits, u_small_int, u_u32) = generate_benchmark_data(m, n);
            
            // Transposition (Excluded from benchmark)
            let u_transposed_bits = transpose_bit_vectors(&u_bits, m, n);
            let u_packed_small_int = transpose_and_pack_small_int_vectors(&u_small_int, m, n);


            // --- Task 1 Setup (Original - Bits) ---
            // M4R Preprocessing for all primes (Excluded from benchmark)
            let tables1 = task1::preprocess_m4r(&a, &PRIMES_TASK1);

            group.bench_with_input(BenchmarkId::new("Task1 (Bits+FullCRT_5P+M4R_8bit)", &input_size), &(m, n), |bencher, &(_m, _n)| {
                bencher.iter(|| {
                    // Benchmark includes residue computation and reconstruction.
                    task1::task1(&b, &u_transposed_bits, m, n, &tables1, &PRIMES_TASK1, &crt_context1)
                });
            });

            // --- Task 1B Setup (New - Small Ints) ---
            // M4R Preprocessing for all primes (Excluded from benchmark)
            let tables1b = task1b::preprocess_m4r(&a, &PRIMES_TASK1);

            group.bench_with_input(BenchmarkId::new("Task1B (SmallInt+FullCRT_5P+M4R_16bit)", &input_size), &(m, n), |bencher, &(_m, _n)| {
                bencher.iter(|| {
                    // Benchmark includes residue computation and reconstruction.
                    task1b::task1b(&b, &u_packed_small_int, m, n, &tables1b, &PRIMES_TASK1, &crt_context1)
                });
            });


            // --- Task 2 Setup ---
            group.bench_with_input(BenchmarkId::new("Task2 (U32+FullCRT_3P)", &input_size), &(m, n), |bencher, &(_m, _n)| {
                bencher.iter(|| {
                    // Benchmark includes residue computation and reconstruction.
                    task2::task2(&b, &u_u32, m, n, &PRIMES_TASK2, &crt_context2)
                });
            });

            // --- Task 3 Setup ---
            let q = get_128bit_prime();
            
            // M4R Preprocessing (Excluded from benchmark)
            let monty_params_setup: Arc<DynResidueParams<LIMBS_U128>> = Arc::new(DynResidueParams::new(&q));
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