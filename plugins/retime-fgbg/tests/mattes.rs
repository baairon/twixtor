//! The FG matte layers of plugin 03 are read as stored: the matte channel is the layer's own channel, not
//! the channel premultiplied by the matte layer's alpha (the original's raw matte plane; the layers grid's
//! `chan1-alpha` and `chan4-alpha` references differ).

mod common;

use common::*;
use fx_core::{Depth, Frame};

/// `n` matte frames of `w x h`, every pixel `[v, v, v, a]`.
fn matte(w: usize, h: usize, v: u8, a: u8, n: usize) -> Vec<Frame> {
    (0..n).map(|_| f8(w, h, &vec![[v, v, v, a]; w * h])).collect()
}

#[test]
fn a_white_matte_channel_over_a_zero_alpha_is_a_white_matte() {
    let (w, h) = (16, 12);
    let clip = random_clip(Depth::U8, w, h, 4, 11);
    let params = [("speed_percent", "50"), ("fg1_matte_chan", "1")];
    let render = |m: Vec<Frame>| Rig::new(clip.clone(), &params).with_layer("fg1_matte", m).render(1).unwrap();
    let stored = render(matte(w, h, 255, 0, 4));
    let opaque = render(matte(w, h, 255, 255, 4));
    let black = render(matte(w, h, 0, 255, 4));
    // the red channel as stored is 255 everywhere, whatever the alpha: the same matte as an opaque white one
    assert_eq!(stored, opaque);
    // and the matte is used: a black red channel gives another frame
    assert_ne!(opaque, black);
}
