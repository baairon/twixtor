//! The triangle scan converter of the textured forward warp, with its edge initialiser.
//!
//! Everything is single precision in the order written here; another order changes the last bits.
//! The three tests against a double constant (`span > 1e-08`, `frac <= 0.01`, `span > 1e-05`)
//! widen the float first.

/// One vertex of the payload `[x, y, u, v, d]`: the position from `mesh::position_row`,
/// `u = c / w1`, `v = r / h1` and `d` the mip level from `mesh::mip_level`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Vtx {
    pub x: f32,
    pub y: f32,
    pub u: f32,
    pub v: f32,
    pub d: f32,
}

/// Receiver of the covered pixels.
pub(crate) trait Sink {
    /// A covered pixel `(x, y)` with the interpolated texture coordinates and mip level.
    fn hit(&mut self, x: usize, y: usize, u: f32, v: f32, d: f32);
    /// The degenerate row pass would write outside the image; nothing is written and the warp
    /// reports `Unsupported`.
    fn outside(&mut self);
}

/// An active edge: its y range, the current `x`, `u`, `v`, `d` and their steps per scanline.
#[derive(Clone, Copy, Default)]
struct Edge {
    ay: f32,
    by: f32,
    x: f32,
    u: f32,
    v: f32,
    d: f32,
    dx: f32,
    du: f32,
    dv: f32,
    dd: f32,
}

/// Truncation toward zero; NaN and out of range values give the integer indefinite `i32::MIN`.
#[inline(always)]
pub(crate) fn cvtt32(x: f32) -> i32 {
    if x.is_nan() || x >= 2_147_483_648.0 || x < -2_147_483_648.0 {
        i32::MIN
    } else {
        x as i32
    }
}

/// Activates the edge `a -> b` when `a.y <= y < b.y`; the new edge goes to `edges[0]` when no edge
/// is active, to `edges[1]` otherwise. The state is initialised for the scanline `y - 1` (the
/// caller advances it once before use).
#[inline]
fn edge_init(a: &Vtx, b: &Vtx, y: f32, edges: &mut [Edge; 2], count: &mut u32) {
    if a.y <= y && y < b.y {
        let slot = if *count != 0 { 1 } else { 0 };
        *count += 1;
        let dx = b.x - a.x;
        let du = b.u - a.u;
        let dv = b.v - a.v;
        let dd = b.d - a.d;
        let inv = 1.0f32 / (b.y - a.y);
        let t = ((y - 1.0f32) - a.y) * inv;
        edges[slot] = Edge {
            ay: a.y,
            by: b.y,
            x: dx * t + a.x,
            u: du * t + a.u,
            v: dv * t + a.v,
            d: dd * t + a.d,
            dx: dx * inv,
            du: du * inv,
            dv: dv * inv,
            dd: dd * inv,
        };
    }
}

/// Sorts the three vertices by `y`. Ties keep the order given, except `v2.y == v0.y < v1.y`,
/// which gives `v2, v0, v1`.
fn sort_by_y(v: &[Vtx; 3]) -> [Vtx; 3] {
    let (y0, y1, y2) = (v[0].y, v[1].y, v[2].y);
    if y1 < y0 {
        if y2 < y0 {
            if y2 < y1 {
                [v[2], v[1], v[0]]
            } else {
                [v[1], v[2], v[0]]
            }
        } else {
            [v[1], v[0], v[2]]
        }
    } else if y2 < y1 {
        if y0 < y2 {
            [v[0], v[2], v[1]]
        } else {
            [v[2], v[0], v[1]]
        }
    } else {
        [v[0], v[1], v[2]]
    }
}

/// One span of the main pass: `l` and `r` are the left and right edge, `y` the scanline.
fn span<S: Sink>(l: &Edge, r: &Edge, y: f32, w1: i32, sink: &mut S) {
    let x0 = cvtt32(l.x.ceil()).max(0);
    let x1 = cvtt32(r.x.floor()).min(w1);
    let wid = r.x - l.x;
    if x0 > x1 {
        return;
    }
    // a span not wider than 1e-08 (compared as a double) is skipped
    if !((wid as f64) > 1e-08f64) {
        return;
    }
    let yi = cvtt32(y) as usize;
    if x0 == x1 {
        let t = ((x0 as f32) - l.x) * (1.0f32 / wid);
        let u = (r.u - l.u) * t + l.u;
        let v = (r.v - l.v) * t + l.v;
        let d = (r.d - l.d) * t + l.d;
        sink.hit(x0 as usize, yi, u, v, d);
    } else {
        let inv = 1.0f32 / wid;
        let du = r.u - l.u;
        let dv = r.v - l.v;
        let dd = r.d - l.d;
        let t = ((x0 as f32) - l.x) * inv;
        let mut u = du * t + l.u;
        let mut v = dv * t + l.v;
        let mut d = dd * t + l.d;
        let mut x = x0;
        while x < x1 {
            sink.hit(x as usize, yi, u, v, d);
            u += du * inv;
            v += dv * inv;
            d += dd * inv;
            x += 1;
        }
        sink.hit(x as usize, yi, u, v, d);
    }
}

/// Scan converts one triangle of the mesh (`FUN_1806a9160`). `w1` and `h1` are the destination
/// `width - 1` and `height - 1`. Rows `y_start..=y_end`, closed spans in x (shared edges are covered by
/// both triangles), edges active for `a.y <= y < b.y`; the last row `h1` is only reached through the
/// degenerate pass at the end.
pub(crate) fn triangle<S: Sink>(tri: &[Vtx; 3], w1: i32, h1: i32, sink: &mut S) {
    let w1f = w1 as f32;
    let h1f = h1 as f32;
    let t = tri;
    // rejection: all x < 0, all x > w1, all y < 0 or all y > h1 (NaN compares false)
    let ok = (0.0 <= t[0].x || 0.0 <= t[1].x || 0.0 <= t[2].x)
        && (0.0 <= t[0].y || 0.0 <= t[1].y || 0.0 <= t[2].y)
        && (t[0].x <= w1f || t[1].x <= w1f || t[2].x <= w1f)
        && (t[0].y <= h1f || t[1].y <= h1f || t[2].y <= h1f);
    if !ok {
        return;
    }
    let vs = sort_by_y(tri);
    let mut y = vs[0].y.ceil();
    let mut y_end = vs[2].y.floor();
    if y < 0.0 {
        y = 0.0;
    }
    if h1f < y_end {
        y_end = h1f;
    }
    if y <= y_end {
        let mut vi = 0usize;
        let mut count = 0u32;
        let mut e = [Edge::default(); 2];
        loop {
            if vi < 3 {
                loop {
                    if y < vs[vi].y {
                        break;
                    }
                    if count != 0 {
                        if count == 1 {
                            if e[0].ay <= y && e[0].by <= y {
                                count = 0;
                            }
                        } else if count == 2 {
                            if y < e[1].ay || y < e[1].by {
                                if !(y < e[0].ay || y < e[0].by) {
                                    // the first edge ended: the second takes its slot
                                    e[0] = e[1];
                                    count = 1;
                                }
                            } else if e[0].ay <= y && e[0].by <= y {
                                count = 0;
                            } else {
                                count = 1;
                            }
                        }
                    }
                    match vi {
                        0 => {
                            edge_init(&vs[0], &vs[1], y, &mut e, &mut count);
                            edge_init(&vs[0], &vs[2], y, &mut e, &mut count);
                        }
                        1 => {
                            edge_init(&vs[1], &vs[2], y, &mut e, &mut count);
                            edge_init(&vs[1], &vs[0], y, &mut e, &mut count);
                        }
                        _ => {
                            edge_init(&vs[2], &vs[0], y, &mut e, &mut count);
                            edge_init(&vs[2], &vs[1], y, &mut e, &mut count);
                        }
                    }
                    vi += 1;
                    if vi >= 3 {
                        break;
                    }
                }
            }
            for k in 0..2 {
                e[k].x += e[k].dx;
                e[k].u += e[k].du;
                e[k].v += e[k].dv;
                e[k].d += e[k].dd;
            }
            if count == 2 {
                // the left edge is the second slot only when it lies strictly left of the first
                if e[1].x < e[0].x {
                    span(&e[1], &e[0], y, w1, sink);
                } else {
                    span(&e[0], &e[1], y, w1, sink);
                }
            }
            y += 1.0f32;
            if !(y <= y_end) {
                break;
            }
        }
    }
    // degenerate pass: both lower vertices exactly on the last row
    let (mid, bot) = (&vs[1], &vs[2]);
    if mid.y == h1f && mid.y == bot.y {
        let (l, r) = if bot.x <= mid.x { (bot, mid) } else { (mid, bot) };
        let yy = l.y;
        let yt = cvtt32(yy) as f32;
        if 0.0 <= yt && yt <= h1f && !(((yy - yt) as f64) > 0.01f64) {
            let x0 = cvtt32(l.x.ceil()).max(0);
            let x1 = cvtt32(r.x.floor()).min(w1);
            let wid = r.x - l.x;
            if (wid as f64) > 1e-05f64 {
                let inv = 1.0f32 / wid;
                let yi = cvtt32(yt);
                let emit = |x: i32, u: f32, v: f32, d: f32, sink: &mut S| {
                    if x < 0 || x > w1 || yi < 0 || yi > h1 {
                        sink.outside();
                    } else {
                        sink.hit(x as usize, yi as usize, u, v, d);
                    }
                };
                if x0 == x1 {
                    let tt = ((x0 as f32) - l.x) * inv;
                    let u = (r.u - l.u) * tt + l.u;
                    let v = (r.v - l.v) * tt + l.v;
                    let d = (r.d - l.d) * tt + l.d;
                    emit(x0, u, v, d, sink);
                } else {
                    let du = r.u - l.u;
                    let dv = r.v - l.v;
                    let dd = r.d - l.d;
                    let tt = ((x0 as f32) - l.x) * inv;
                    let mut u = du * tt + l.u;
                    let mut v = dv * tt + l.v;
                    let mut d = dd * tt + l.d;
                    let mut x = x0;
                    while x < x1 {
                        emit(x, u, v, d, sink);
                        u += du * inv;
                        v += dv * inv;
                        d += dd * inv;
                        x += 1;
                    }
                    // no `x0 <= x1` test here: a span narrower than a pixel still writes `x0`
                    emit(x, u, v, d, sink);
                }
            }
        }
    }
}
