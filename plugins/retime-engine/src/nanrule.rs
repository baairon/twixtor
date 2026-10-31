//! Which NaN an arithmetic operation returns when both of its operands are NaNs.
//!
//! Numbers never depend on this module (every function returns the plain IEEE result for them, bit
//! for bit). It only fixes the payload and the sign of a NaN result of the single precision sums,
//! differences, products and quotients of the 32 bits per channel paths, where the special value
//! test clips put NaNs of both signs into one operation.
//!
//! Two rules exist for an operation `dest = dest op src` with two NaN operands:
//!
//! * **Processor rule** (Intel SDM vol. 1, table 4-7, SSE column): the first source operand (the
//!   destination register of the instruction) is returned, quieted. A NaN in only one operand is
//!   returned, quieted. An invalid operation (`inf - inf`, `0 * inf`, `0 / 0`) gives the default
//!   NaN `0xffc00000`.
//! * **Emulator rule** (the x87 selection of QEMU's soft float, which Unicorn 2.1.4 applies to the
//!   SSE instructions too): of two QNaNs the one with the larger fraction wins, with equal
//!   fractions the one with the positive sign; a QNaN beats a SNaN; of two SNaNs the larger
//!   fraction wins; the result is quieted. It does not depend on the operand order. MEASURED in
//!   the emulator the references were rendered with (`addss`, `subss`, `mulss`, `divss`, `addps`
//!   with `0xffc00000` against `0x7fc00000` give `0x7fc00000` in both operand orders), and READ in the
//!   original's final blend: the references of every special value
//!   clip carry this rule, not the processor's.
//!
//! The default (feature `emulator-nan`, on) is the emulator rule because the references the
//! rewrite is graded against were rendered under that emulator; the processor rule is what
//! the plugin does on real hardware (build without the default features of `retime-engine`).

/// The quiet bit of a single precision NaN.
const QUIET: u32 = 0x0040_0000;

/// `x` with the quiet bit set (payload and sign kept): what an arithmetic instruction returns for a NaN operand.
#[inline(always)]
pub fn quiet(x: f32) -> f32 {
    f32::from_bits(x.to_bits() | QUIET)
}

#[inline(always)]
fn is_signalling(x: f32) -> bool {
    x.to_bits() & QUIET == 0
}

/// The NaN an operation returns when both operands are NaNs. `first` is the destination operand of the instruction
/// (only the processor rule looks at the order).
#[inline(always)]
pub fn pick(first: f32, second: f32) -> f32 {
    #[cfg(feature = "emulator-nan")]
    {
        let (ba, bb) = (first.to_bits(), second.to_bits());
        let (sa, sb) = (is_signalling(first), is_signalling(second));
        let (fa, fb) = (ba & 0x007f_ffff, bb & 0x007f_ffff);
        // a is chosen when its fraction is larger, or equal with a positive a and a negative b
        let a_larger = fa > fb || (fa == fb && (ba >> 31) < (bb >> 31));
        let take_b = match (sa, sb) {
            (true, true) | (false, false) => !a_larger,
            (true, false) => true,
            (false, true) => false,
        };
        quiet(if take_b { second } else { first })
    }
    #[cfg(not(feature = "emulator-nan"))]
    {
        let _ = second;
        quiet(first)
    }
}

/// The result `r` of `first op second` with the NaN rule applied: numbers and invalid operations unchanged.
#[inline(always)]
fn fix(r: f32, first: f32, second: f32) -> f32 {
    if r.is_nan() {
        if first.is_nan() {
            if second.is_nan() {
                pick(first, second)
            } else {
                quiet(first)
            }
        } else if second.is_nan() {
            quiet(second)
        } else {
            r
        }
    } else {
        r
    }
}

/// `first + second`.
#[inline(always)]
pub fn add(first: f32, second: f32) -> f32 {
    fix(first + second, first, second)
}

/// `first - second`.
#[inline(always)]
pub fn sub(first: f32, second: f32) -> f32 {
    fix(first - second, first, second)
}

/// `first * second`.
#[inline(always)]
pub fn mul(first: f32, second: f32) -> f32 {
    fix(first * second, first, second)
}

/// `first / second`.
#[inline(always)]
pub fn div(first: f32, second: f32) -> f32 {
    fix(first / second, first, second)
}

/// The quiet bit of a double precision NaN.
const QUIET64: u64 = 0x0008_0000_0000_0000;

/// [`pick`] for double precision operands (`addsd`, `mulsd` and the other scalar double instructions follow the same
/// selection: the emulator compares the 52 bit fractions, the processor returns the first operand).
#[inline(always)]
pub fn pick64(first: f64, second: f64) -> f64 {
    #[cfg(feature = "emulator-nan")]
    {
        let (ba, bb) = (first.to_bits(), second.to_bits());
        let (sa, sb) = (ba & QUIET64 == 0, bb & QUIET64 == 0);
        let (fa, fb) = (ba & 0x000f_ffff_ffff_ffff, bb & 0x000f_ffff_ffff_ffff);
        let a_larger = fa > fb || (fa == fb && (ba >> 63) < (bb >> 63));
        let take_b = match (sa, sb) {
            (true, true) | (false, false) => !a_larger,
            (true, false) => true,
            (false, true) => false,
        };
        f64::from_bits(if take_b { bb } else { ba } | QUIET64)
    }
    #[cfg(not(feature = "emulator-nan"))]
    {
        let _ = second;
        f64::from_bits(first.to_bits() | QUIET64)
    }
}

/// `first + second` in double precision with the NaN rule (numbers and invalid operations unchanged).
#[inline(always)]
pub fn add64(first: f64, second: f64) -> f64 {
    let r = first + second;
    if !r.is_nan() {
        r
    } else if first.is_nan() && second.is_nan() {
        pick64(first, second)
    } else if first.is_nan() {
        f64::from_bits(first.to_bits() | QUIET64)
    } else if second.is_nan() {
        f64::from_bits(second.to_bits() | QUIET64)
    } else {
        r
    }
}

/// A set of arithmetic operations for kernels written once and run in two ways: [`Plain`] is the processor's own (fast, and
/// vectorisable), [`Rule`] applies the NaN rule of this module. A kernel runs as `Plain` and is repeated as `Rule` only when
/// a result holds a NaN ([`any_nan`]): without a NaN operand both give the same bits, and a NaN needs a NaN operand or an
/// invalid operation, whose default NaN is the same in both.
pub trait Arith {
    fn add(first: f32, second: f32) -> f32;
    fn sub(first: f32, second: f32) -> f32;
    fn mul(first: f32, second: f32) -> f32;
    fn div(first: f32, second: f32) -> f32;
}

/// The processor's own operations.
pub struct Plain;

impl Arith for Plain {
    #[inline(always)]
    fn add(first: f32, second: f32) -> f32 {
        first + second
    }
    #[inline(always)]
    fn sub(first: f32, second: f32) -> f32 {
        first - second
    }
    #[inline(always)]
    fn mul(first: f32, second: f32) -> f32 {
        first * second
    }
    #[inline(always)]
    fn div(first: f32, second: f32) -> f32 {
        first / second
    }
}

/// The operations with the NaN rule of this module.
pub struct Rule;

impl Arith for Rule {
    #[inline(always)]
    fn add(first: f32, second: f32) -> f32 {
        add(first, second)
    }
    #[inline(always)]
    fn sub(first: f32, second: f32) -> f32 {
        sub(first, second)
    }
    #[inline(always)]
    fn mul(first: f32, second: f32) -> f32 {
        mul(first, second)
    }
    #[inline(always)]
    fn div(first: f32, second: f32) -> f32 {
        div(first, second)
    }
}

/// True when one of the values is a NaN.
#[inline(always)]
pub fn any_nan(v: &[f32]) -> bool {
    v.iter().any(|x| x.is_nan())
}

/// [`lerp`] over a set of operations.
#[inline(always)]
pub fn lerp_g<A: Arith>(a: f32, b: f32, f: f32) -> f32 {
    A::add(A::mul(A::sub(b, a), f), a)
}

/// [`lerp_base_first`] over a set of operations.
#[inline(always)]
pub fn lerp_base_first_g<A: Arith>(a: f32, b: f32, f: f32) -> f32 {
    A::add(a, A::mul(A::sub(b, a), f))
}

/// `(b - a) * f + a` as the final blend computes it (READ): `d = b - a`, `m = d * f`, `m + a`,
/// the product being the first operand of the sum (only the processor rule looks at the order).
#[inline(always)]
pub fn lerp(a: f32, b: f32, f: f32) -> f32 {
    add(mul(sub(b, a), f), a)
}

/// The same lerp as the bilinear sampler computes it: the base `a` is the first operand of the sum (MEASURED, the
/// processor rule only; the emulator rule does not look at the order).
#[inline(always)]
pub fn lerp_base_first(a: f32, b: f32, f: f32) -> f32 {
    add(a, mul(sub(b, a), f))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: f32) -> u32 {
        x.to_bits()
    }
    const POS: u32 = 0x7fc0_0000;
    const NEG: u32 = 0xffc0_0000;

    #[test]
    fn numbers_are_the_plain_ieee_result() {
        for (a, b) in [(1.0f32, 3.0f32), (-0.0, 0.0), (2.5, -7.0), (f32::MAX, f32::INFINITY), (1.0e-40, 3.0e-40)] {
            assert_eq!(bits(add(a, b)), bits(a + b));
            assert_eq!(bits(sub(a, b)), bits(a - b));
            assert_eq!(bits(mul(a, b)), bits(a * b));
            assert_eq!(bits(div(a, b)), bits(a / b));
        }
        for (a, b, f) in [(1.0f32, 3.0f32, 0.25f32), (-0.0, 0.0, 0.5), (2.5, -7.0, 0.75), (f32::MAX, f32::INFINITY, 0.5)] {
            assert_eq!(bits(lerp(a, b, f)), bits((b - a) * f + a));
        }
    }

    #[test]
    fn plain_and_rule_agree_on_numbers() {
        let vals = [0.0f32, -0.0, 1.0, -2.5, 3.0e-40, f32::MAX, f32::INFINITY, f32::NEG_INFINITY];
        for &a in &vals {
            for &b in &vals {
                for &f in &[0.25f32, 0.5, 1.0] {
                    let (p, r) = (lerp_g::<Plain>(a, b, f), lerp_g::<Rule>(a, b, f));
                    assert!(bits(p) == bits(r) || (p.is_nan() && r.is_nan()), "{a} {b} {f}");
                    let (p, r) = (lerp_base_first_g::<Plain>(a, b, f), lerp_base_first_g::<Rule>(a, b, f));
                    assert!(bits(p) == bits(r) || (p.is_nan() && r.is_nan()), "{a} {b} {f}");
                }
            }
        }
        assert!(any_nan(&[1.0, f32::NAN]) && !any_nan(&[1.0, 2.0]));
    }

    #[test]
    fn one_nan_operand_is_returned_quieted() {
        let sig = f32::from_bits(0x7f80_0001);
        assert_eq!(bits(add(f32::from_bits(NEG), 1.0)), NEG);
        assert_eq!(bits(add(1.0, f32::from_bits(POS))), POS);
        assert_eq!(bits(mul(sig, 2.0)), 0x7fc0_0001);
        assert_eq!(bits(div(2.0, f32::from_bits(NEG))), NEG);
    }

    #[test]
    fn invalid_operations_give_the_default_nan() {
        // black_box: a constant folded NaN would be the positive one, the processor makes the negative default NaN
        let (inf, zero) = (std::hint::black_box(f32::INFINITY), std::hint::black_box(0.0f32));
        assert_eq!(bits(sub(inf, inf)), NEG);
        assert_eq!(bits(mul(zero, inf)), NEG);
        assert_eq!(bits(div(zero, zero)), NEG);
    }

    #[cfg(feature = "emulator-nan")]
    #[test]
    fn two_nans_follow_the_emulator_rule() {
        let (pos, neg) = (f32::from_bits(POS), f32::from_bits(NEG));
        // equal fractions: the positive one, whatever the order
        for (x, y) in [(pos, neg), (neg, pos)] {
            assert_eq!(bits(add(x, y)), POS);
            assert_eq!(bits(sub(x, y)), POS);
            assert_eq!(bits(mul(x, y)), POS);
            assert_eq!(bits(div(x, y)), POS);
        }
        assert_eq!(bits(add(neg, neg)), NEG);
        // the larger fraction wins, whatever the sign and the order
        let big = f32::from_bits(0xffc0_0002);
        let small = f32::from_bits(0x7fc0_0001);
        assert_eq!(bits(add(big, small)), 0xffc0_0002);
        assert_eq!(bits(add(small, big)), 0xffc0_0002);
        // a QNaN beats a SNaN, the winner stays quiet
        let sig = f32::from_bits(0x7fa0_0000);
        assert_eq!(bits(mul(sig, neg)), NEG);
        assert_eq!(bits(mul(neg, sig)), NEG);
        // two SNaNs: the larger fraction, quieted
        let sig2 = f32::from_bits(0x7f80_0007);
        assert_eq!(bits(add(sig, sig2)), 0x7fe0_0000);
        // the lerp of the sampler: base NaN against far NaN
        assert_eq!(bits(lerp(neg, pos, 0.5)), POS);
        assert_eq!(bits(lerp(pos, neg, 0.5)), POS);
    }

    #[test]
    fn double_sums_of_numbers_are_the_plain_ieee_result() {
        for (a, b) in [(1.0f64, 3.0f64), (-0.0, 0.0), (2.5, -7.0), (f64::MAX, f64::INFINITY), (1.0e-310, 3.0e-310)] {
            assert_eq!(add64(a, b).to_bits(), (a + b).to_bits());
        }
        let inf = std::hint::black_box(f64::INFINITY);
        assert_eq!(add64(inf, -inf).to_bits(), 0xfff8_0000_0000_0000);
        assert_eq!(add64(f64::from_bits(0xfff8_0000_0000_0000), 1.0).to_bits(), 0xfff8_0000_0000_0000);
    }

    #[cfg(feature = "emulator-nan")]
    #[test]
    fn two_double_nans_follow_the_emulator_rule() {
        let (pos, neg) = (f64::from_bits(0x7ff8_0000_0000_0000), f64::from_bits(0xfff8_0000_0000_0000));
        for (x, y) in [(pos, neg), (neg, pos)] {
            assert_eq!(add64(x, y).to_bits(), 0x7ff8_0000_0000_0000);
        }
        assert_eq!(add64(neg, neg).to_bits(), 0xfff8_0000_0000_0000);
        let big = f64::from_bits(0xfff8_0000_0000_0002);
        assert_eq!(add64(big, pos).to_bits(), 0xfff8_0000_0000_0002);
        assert_eq!(add64(pos, big).to_bits(), 0xfff8_0000_0000_0002);
        let sig = f64::from_bits(0x7ff4_0000_0000_0000);
        assert_eq!(add64(sig, neg).to_bits(), 0xfff8_0000_0000_0000);
        assert_eq!(add64(sig, 1.0).to_bits(), 0x7ffc_0000_0000_0000);
    }

    #[cfg(not(feature = "emulator-nan"))]
    #[test]
    fn two_nans_follow_the_processor_rule() {
        let (pos, neg) = (f32::from_bits(POS), f32::from_bits(NEG));
        assert_eq!(bits(add(neg, pos)), NEG);
        assert_eq!(bits(add(pos, neg)), POS);
        assert_eq!(bits(sub(pos, neg)), POS);
    }
}
