"""
Reference model of the optical flow solver (numpy, float32), used to generate the differential fixtures
of tests/flow_d.rs (see gen_flow_fixtures.py). It checks sign conventions and structure on synthetic
data. It is not bit exact (numpy vectorised order of operations).

usage: python -I flow_ref.py
"""
import numpy as np

F = np.float32


def gauss_kernel(r):
    r = F(r)
    n = int(r) + 1
    s3 = r / F(3.0)
    d = s3 * s3
    k = np.array([np.power(F(2.0), -(F(i) * F(i)) / d) for i in range(n)], dtype=F)
    den = F(k.sum(dtype=F) * F(2.0) - k[0])
    return (k / den).astype(F)


def conv_line_axis(img, k, axis):
    """edge renormalised symmetric convolution along an axis."""
    n = len(k)
    a = np.moveaxis(img, axis, 0)
    L = a.shape[0]
    out = np.zeros_like(a, dtype=F)
    for i in range(L):
        num = k[0] * a[i]
        den = k[0]
        for j in range(n - 1, 0, -1):
            if i + j < L:
                num = num + k[j] * a[i + j]
                den = den + k[j]
            if i - j >= 0:
                num = num + k[j] * a[i - j]
                den = den + k[j]
        out[i] = num / den
    return np.moveaxis(out, 0, axis)


def blur(img, r):
    k = gauss_kernel(r)
    t = conv_line_axis(img, k, 1)   # horizontal (x is axis 1)
    return conv_line_axis(t, k, 0)  # vertical


def bilinear(img, xs, ys):
    h, w = img.shape[:2]
    ix = np.trunc(xs).astype(np.int64)
    iy = np.trunc(ys).astype(np.int64)
    fx = (xs - ix).astype(F)
    fy = (ys - iy).astype(F)
    m = ix >= w - 1
    ix = np.where(m, w - 2, ix)
    fx = np.where(m, F(1.0), fx)
    m = iy >= h - 1
    iy = np.where(m, h - 2, iy)
    fy = np.where(m, F(1.0), fy)
    ix = np.clip(ix, 0, w - 2)
    iy = np.clip(iy, 0, h - 2)
    p00 = img[iy, ix]; p10 = img[iy, ix + 1]; p01 = img[iy + 1, ix]; p11 = img[iy + 1, ix + 1]
    top = (p10 - p00) * fx + p00
    bot = (p11 - p01) * fx + p01
    return (bot - top) * fy + top


def calc_grad(I):
    h, w = I.shape
    gx = np.zeros((h, w), F); gy = np.zeros((h, w), F)
    for y in range(h):
        for x in range(w):
            n = F(0); d = F(0)
            if x >= 1: n = (I[y, x] - I[y, x - 1]) + F(0); d = F(1)
            if x + 1 <= w - 1: n = n + (I[y, x + 1] - I[y, x]); d = d + F(1)
            gx[y, x] = n / d
            n = F(0); d = F(0)
            if y >= 1: n = (I[y, x] - I[y - 1, x]) + F(0); d = F(1)
            if y + 1 <= h - 1: n = n + (I[y + 1, x] - I[y, x]); d = d + F(1)
            gy[y, x] = n / d
    return np.stack([gx, gy], -1)


def calc_b(I1, I0, grad, flow):
    h, w = I1.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(F)
    sx = xx - flow[..., 0]
    sy = yy - flow[..., 1]
    ok = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    i0 = bilinear(I0, np.where(ok, sx, 0), np.where(ok, sy, 0))
    r = (i0 - I1)[..., None] * grad
    return np.where(ok[..., None], r, F(0)).astype(F)


def new_disp(flow, A, B):
    a = A[..., 0]; b = A[..., 1]; d = A[..., 2]
    t = b * b - d * a
    m = np.abs(t) > F(40.0)
    tt = np.where(m, t, F(1.0))
    ux = (b * B[..., 1] - d * B[..., 0]) / tt + flow[..., 0]
    uy = (b * B[..., 0] - a * B[..., 1]) / tt + flow[..., 1]
    out = flow.copy()
    out[..., 0] = np.where(m, ux, flow[..., 0])
    out[..., 1] = np.where(m, uy, flow[..., 1])
    return out


def resample(src, w, h):
    sh, sw = src.shape[:2]
    sx = F(sw - 1) / F(w - 1); sy = F(sh - 1) / F(h - 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(F)
    return bilinear(src, xx * sx, yy * sy)


def build_pyramid(luma, levels):
    lv = [luma.astype(F)]
    sizes = [luma.shape]
    h, w = luma.shape
    for i in range(1, levels):
        w = max(2, w // 2); h = max(2, h // 2); sizes.append((h, w))
    k5 = np.array([0.375, 0.25, 0.0625], F)
    orig = [luma.astype(F)]
    out = []
    for i in range(levels):
        tmp = conv_line_axis(conv_line_axis(orig[i], k5, 1), k5, 0)
        out.append((orig[i] - tmp).astype(F))
        if i + 1 < levels:
            orig.append(resample(tmp, sizes[i + 1][1], sizes[i + 1][0]))
    return out


def upsample(flow, w, h):
    sh, sw = flow.shape[:2]
    sx = F(w - 1) / F(sw - 1); sy = F(h - 1) / F(sh - 1)
    rx = F(1.0 / float(sx)); ry = F(1.0 / float(sy))
    yy, xx = np.mgrid[0:h, 0:w].astype(F)
    u = bilinear(flow[..., 0], xx * rx, yy * ry) * sx
    v = bilinear(flow[..., 1], xx * rx, yy * ry) * sy
    return np.stack([u, v], -1).astype(F)


def solve(X, Y, sensitivity=1.0, iters=3):
    """returns out with X(x) ~ Y(x + out(x))"""
    levels = 1
    h, w = X.shape
    while True:
        w //= 2; h //= 2
        if w >= 2 and h >= 2 and w > 2 and h > 2 and not (w == 2 and h == 2):
            levels += 1
        else:
            break
    s = F(sensitivity)
    x = F(np.log(np.power(s, F(2.0))) / np.log(F(2.0))) + F(levels)
    if x >= 0:
        if x < levels:
            start = int(np.floor(x)); damp = F(x - np.floor(x))
        else:
            start = levels - 1; damp = F(1.0)
    else:
        start = 0; damp = F(0.0)
    A_ = build_pyramid(X, levels)
    B_ = build_pyramid(Y, levels)
    flow = np.zeros(A_[start].shape + (2,), F)
    for l in range(start, -1, -1):
        I1 = A_[l]; I0 = B_[l]
        grad = calc_grad(I1)
        A = np.stack([grad[..., 0] * grad[..., 0], grad[..., 0] * grad[..., 1], grad[..., 1] * grad[..., 1]], -1)
        A = blur(A, 7.0)
        for it in range(iters):
            B = calc_b(I1, I0, grad, flow)
            B = blur(B, 7.0)
            flow = new_disp(flow, A, B)
            flow = blur(flow, 5.0)
        if l == start and damp != 1.0:
            flow = flow * damp
        if l > 0:
            h2, w2 = A_[l - 1].shape
            flow = upsample(flow, w2, h2)
    return -flow


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    base = rng.random((96, 96)).astype(F) * F(255.0)
    base = blur(base, 3.0)
    for shift in [(3, 2), (-4, 1), (0, -3)]:
        sx, sy = shift
        h, w = base.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(F)
        # Y(x) = X(x - s)  =>  X(x) = Y(x + s)  => expected out = +s
        Y = bilinear(base, np.clip(xx - sx, 0, w - 2), np.clip(yy - sy, 0, h - 2))
        out = solve(base, Y, 1.0)
        c = out[24:72, 24:72].reshape(-1, 2)
        print("shift", shift, "mean flow", c.mean(0), "median", np.median(c, 0))
