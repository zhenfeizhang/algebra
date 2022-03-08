extern crate criterion;

use ark_ec::{msm::VariableBase, short_weierstrass_jacobian::GroupAffine, SWModelParameters};
use ark_ff::{FftField, PrimeField};
use ark_poly::{
    polynomial::{univariate::DensePolynomial, UVPolynomial},
    EvaluationDomain, Evaluations, MixedRadixEvaluationDomain, Radix2EvaluationDomain,
};
use ark_poly_benches::size_range;
use ark_std::{test_rng, UniformRand};
use ark_test_curves::{
    bls12_381::g1::Parameters as bls12_381_g1param, bls12_381::Fr as bls12_381_fr,
    mnt4_753::Fq as mnt6_753_fr,
};
use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};

// degree bounds to benchmark on
// e.g. degree bound of 2^{15}, means we do an FFT for a degree (2^{15} - 1)
// polynomial
const BENCHMARK_MIN_DEGREE: usize = 1 << 15;
const BENCHMARK_MAX_DEGREE_BLS12_381: usize = 1 << 25;
const BENCHMARK_MAX_DEGREE_MNT6_753: usize = 1 << 17;
const BENCHMARK_LOG_INTERVAL_DEGREE: usize = 1;

const ENABLE_RADIX2_BENCHES: bool = true;
const ENABLE_MIXED_RADIX_BENCHES: bool = false;

// returns vec![2^{min}, 2^{min + interval}, ..., 2^{max}], where:
// interval = BENCHMARK_LOG_INTERVAL_DEGREE
// min      = ceil(log_2(BENCHMARK_MIN_DEGREE))
// max      = ceil(log_2(BENCHMARK_MAX_DEGREE))
fn default_size_range_bls12_381() -> Vec<usize> {
    size_range(
        BENCHMARK_LOG_INTERVAL_DEGREE,
        BENCHMARK_MIN_DEGREE,
        BENCHMARK_MAX_DEGREE_BLS12_381,
    )
}

fn default_size_range_mnt6_753() -> Vec<usize> {
    size_range(
        BENCHMARK_LOG_INTERVAL_DEGREE,
        BENCHMARK_MIN_DEGREE,
        BENCHMARK_MAX_DEGREE_MNT6_753,
    )
}

fn setup_bench(
    c: &mut Criterion,
    name: &str,
    bench_fn: fn(&mut Bencher, &usize),
    size_range: &[usize],
) {
    let mut group = c.benchmark_group(name);
    for degree in size_range.iter() {
        group.bench_with_input(BenchmarkId::from_parameter(degree), degree, bench_fn);
    }
    group.finish();
}

fn setup_msm_bench<F, P>(
    c: &mut Criterion,
    name: &str,
    bench_fn: fn(&mut Bencher, &usize),
    size_range: &[usize],
    scalars: &[<P::ScalarField as PrimeField>::BigInt],
    basis: &[GroupAffine<P>],
) where
    F: PrimeField,
    P: SWModelParameters<ScalarField = F>,
{
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    for degree in size_range.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(degree),
            &(degree,
            scalars,
            basis),
            bench_fn,
        );
    }
    group.finish();
}

fn fft_common_setup<F: FftField, D: EvaluationDomain<F>>(degree: usize) -> (D, Vec<F>) {
    let mut rng = &mut ark_std::test_rng();
    let domain = D::new(degree).unwrap();
    let a = DensePolynomial::<F>::rand(degree - 1, &mut rng)
        .coeffs()
        .to_vec();
    (domain, a)
}

fn bench_fft_root_gen<F: FftField, D: EvaluationDomain<F>>(b: &mut Bencher, degree: &usize) {
    // Per benchmark setup
    let (domain, _) = fft_common_setup::<F, D>(*degree);
    b.iter(|| {
        // Per benchmark iteration
        domain.roots_of_unity();
    });
}

fn bench_fft_in_place<F: FftField, D: EvaluationDomain<F>>(b: &mut Bencher, degree: &usize) {
    // Per benchmark setup
    let (domain, mut a) = fft_common_setup::<F, D>(*degree);
    b.iter(|| {
        // Per benchmark iteration
        domain.fft_in_place(&mut a);
    });
}

fn bench_ifft_in_place<F: FftField, D: EvaluationDomain<F>>(b: &mut Bencher, degree: &usize) {
    // Per benchmark setup
    let (domain, mut a) = fft_common_setup::<F, D>(*degree);
    b.iter(|| {
        // Per benchmark iteration
        domain.ifft_in_place(&mut a);
    });
}

fn bench_coset_fft_in_place<F: FftField, D: EvaluationDomain<F>>(b: &mut Bencher, degree: &usize) {
    // Per benchmark setup
    let (domain, mut a) = fft_common_setup::<F, D>(*degree);
    b.iter(|| {
        // Per benchmark iteration
        domain.coset_fft_in_place(&mut a);
    });
}

fn bench_coset_ifft_in_place<F: FftField, D: EvaluationDomain<F>>(b: &mut Bencher, degree: &usize) {
    // Per benchmark setup
    let (domain, mut a) = fft_common_setup::<F, D>(*degree);
    b.iter(|| {
        // Per benchmark iteration
        domain.coset_ifft_in_place(&mut a);
    });
}

fn bench_interpolation<F: FftField, D: EvaluationDomain<F>>(b: &mut Bencher, degree: &usize) {
    // Per benchmark setup
    let mut rng = test_rng();
    let (domain, _) = fft_common_setup::<F, D>(*degree);
    let evals: Vec<F> = (0..*degree).map(|_| F::rand(&mut rng)).collect();
    let eval = Evaluations::from_vec_and_domain(evals, domain);
    b.iter(|| {
        // Per benchmark iteration
        eval.clone().interpolate();
    });
}

fn bench_msm<F, P>(
    b: &mut Bencher,
    degree: &usize,
    scalars: &[<P::ScalarField as PrimeField>::BigInt],
    basis: &[GroupAffine<P>],
) where
    F: PrimeField,
    P: SWModelParameters<ScalarField = F>,
{
    let scalar_repr = scalars[0..*degree].to_vec();
    let bases = basis[0..*degree].to_vec();

    b.iter(|| {
        // Per benchmark iteration
        let _ = VariableBase::msm(&bases, &scalar_repr);
    });
}

fn fft_benches<F: PrimeField, D: EvaluationDomain<F>>(
    c: &mut Criterion,
    name: &'static str,
    size_range: &[usize],
) {
    let cur_name = format!("{:?} - root_of_unity_generation", name.clone());
    setup_bench(c, &cur_name, bench_fft_root_gen::<F, D>, size_range);
    let cur_name = format!("{:?} - subgroup_fft_in_place", name.clone());
    setup_bench(c, &cur_name, bench_fft_in_place::<F, D>, size_range);
    let cur_name = format!("{:?} - subgroup_ifft_in_place", name.clone());
    setup_bench(c, &cur_name, bench_ifft_in_place::<F, D>, size_range);
    let cur_name = format!("{:?} - coset_fft_in_place", name.clone());
    setup_bench(c, &cur_name, bench_coset_fft_in_place::<F, D>, size_range);
    let cur_name = format!("{:?} - coset_ifft_in_place", name.clone());
    setup_bench(c, &cur_name, bench_coset_ifft_in_place::<F, D>, size_range);
    let cur_name = format!("{:?} - interpolation", name.clone());
    setup_bench(c, &cur_name, bench_interpolation::<F, D>, size_range);
}

fn msm_benches<F, P>(c: &mut Criterion, name: &'static str, size_range: &[usize])
where
    F: PrimeField,
    P: SWModelParameters<ScalarField = F>,
{
    // Per benchmark setup
    let mut rng = test_rng();
    let scalars: Vec<P::ScalarField> = (0..1 << BENCHMARK_MAX_DEGREE_BLS12_381)
        .map(|_| P::ScalarField::rand(&mut rng))
        .collect();
    let scalars_repr: Vec<<P::ScalarField as PrimeField>::BigInt> =
        scalars.iter().map(|x| (*x).into_bigint()).collect();
    let bases: Vec<GroupAffine<P>> = (0..1 << BENCHMARK_MAX_DEGREE_BLS12_381)
        .map(|_| GroupAffine::<P>::rand(&mut rng))
        .collect();

    let cur_name = format!("{:?} - msm", name.clone());
    setup_msm_bench(
        c,
        &cur_name,
        bench_msm::<F, P>,
        size_range,
        &scalars_repr,
        &bases,
    );
}

fn bench_bls12_381(c: &mut Criterion) {
    let name = "bls12_381 - radix2";
    if ENABLE_RADIX2_BENCHES {
        fft_benches::<bls12_381_fr, Radix2EvaluationDomain<bls12_381_fr>>(
            c,
            name,
            &default_size_range_bls12_381(),
        );
    }
    msm_benches::<bls12_381_fr, bls12_381_g1param>(c, name, &default_size_range_bls12_381());
}

fn bench_mnt6_753(c: &mut Criterion) {
    let name = "mnt6_753 - mixed radix";
    if ENABLE_MIXED_RADIX_BENCHES {
        fft_benches::<mnt6_753_fr, MixedRadixEvaluationDomain<mnt6_753_fr>>(
            c,
            name,
            &default_size_range_mnt6_753(),
        );
    }
}

criterion_group!(benches, bench_bls12_381, bench_mnt6_753);
criterion_main!(benches);
