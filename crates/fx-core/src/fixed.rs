//! 16.16 fixed point numbers.
//!
//! Some sliders reach an effect as a 16.16 fixed point integer, so a front-end has to quantise a
//! sampled value the same way before using it.

/// Converts to 16.16 fixed point: scale by 65536, add 0.5 away from zero, truncate. So halves round
/// away from zero.
pub fn float_to_fixed(f: f64) -> i32 {
    (f * 65536.0 + if f < 0.0 { -0.5 } else { 0.5 }) as i32
}

/// Converts from 16.16 fixed point to a float: divide by 65536.
pub fn fixed_to_float(x: i32) -> f64 {
    x as f64 / 65536.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_numbers_and_the_speed_scale() {
        assert_eq!(float_to_fixed(100.0), 6_553_600);
        assert_eq!(float_to_fixed(0.0), 0);
        assert_eq!(float_to_fixed(-1.5), -98_304);
        assert_eq!(fixed_to_float(6_553_600), 100.0);
    }

    #[test]
    fn rounds_to_nearest_with_halves_away_from_zero() {
        assert_eq!(float_to_fixed(33.3), 2_182_349);
        assert_eq!(float_to_fixed(0.5 / 65536.0), 1);
        assert_eq!(float_to_fixed(-0.5 / 65536.0), -1);
        assert_eq!(float_to_fixed(0.4 / 65536.0), 0);
        assert_eq!(float_to_fixed(-0.4 / 65536.0), 0);
    }

    #[test]
    fn multiples_of_the_step_survive() {
        for x in [-70_000, -1, 0, 1, 12_345, 6_553_601] {
            assert_eq!(float_to_fixed(fixed_to_float(x)), x);
        }
    }
}
