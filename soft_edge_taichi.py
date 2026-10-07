"""Optional Taichi backend of soft_edge_mask: exact 3-D squared Euclidean distance transform.

Imported only when a Taichi backend is requested; the rest of DeepMASC does not need Taichi.

Separable algorithm: Meijster, Roerdink and Hesselink's second scan (a lower envelope of parabolas with
integer arithmetic), applied once along each axis. Every line of an axis is independent, so the kernels
parallelise over lines. Each line copies its values into per-line scratch (stack s, boundaries t, input
copy g); lines are processed in batches so the scratch stays bounded whatever the box size.
Values are int32 squared distances, exact. Absent features are INF = 2**24 (above any real squared
distance for boxes up to ~2000 voxels per side, below int32 overflow in the Sep numerator).
"""
import numpy as np
import taichi as ti

INF = 1 << 24
SCRATCH_BYTES = 256 * 1024 * 1024  # budget for the three int32 scratch arrays of one batch
_cpu_layout = True  # scratch is (B, L) on CPU (a line contiguous), (L, B) on CUDA (lines coalesced)
_arch = None  # "cpu" or "cuda" once Taichi is initialised


def _cuda_is_current():
    return ti.lang.impl.current_cfg().arch == ti.cuda


def cuda_available():
    """True when Taichi can run on a CUDA device (initialises Taichi on the CPU if it is not yet)."""
    if _arch is not None:
        return _arch == "cuda"
    try:
        ti.init(arch=ti.cuda, log_level=ti.ERROR)
        return _cuda_is_current()
    except Exception:
        return False
    finally:
        ti.reset()


def init(arch="auto", threads=None):
    """Initialise Taichi once per process and return the architecture in use, "cpu" or "cuda".

    arch: "cpu", "cuda" (raises RuntimeError when no CUDA device is usable) or "auto" (CUDA when there is
    one, else CPU). A later call returns the architecture already in use, and raises if it asks for the other one. threads limits the CPU threads.
    """
    global _cpu_layout, _arch
    if arch not in ("cpu", "cuda", "auto"):
        raise ValueError(f"arch must be 'cpu', 'cuda' or 'auto', not {arch!r}")
    if _arch is not None:
        if arch not in ("auto", _arch):
            raise RuntimeError(f"Taichi is already initialised on {_arch}, cannot switch to {arch} in this process")
        return _arch
    kw = dict(default_fp=ti.f64, default_ip=ti.i32, log_level=ti.ERROR)
    if arch in ("cuda", "auto"):
        try:
            ti.init(arch=ti.cuda, **kw)
            on_cuda = _cuda_is_current()
        except Exception:
            on_cuda = False
        if on_cuda:
            _arch, _cpu_layout = "cuda", False
            return _arch
        ti.reset()
        if arch == "cuda":
            raise RuntimeError("Taichi found no usable CUDA device")
    if threads:
        kw["cpu_max_num_threads"] = int(threads)
    ti.init(arch=ti.cpu, **kw)
    _arch, _cpu_layout = "cpu", True
    return _arch


@ti.func
def _rd(a: ti.template(), b, k):
    return a[b, k] if ti.static(_cpu_layout) else a[k, b]


@ti.func
def _wr(a: ti.template(), b, k, v):
    if ti.static(_cpu_layout):
        a[b, k] = v
    else:
        a[k, b] = v


@ti.func
def _voxel(axis: ti.template(), p, q, k):
    if ti.static(axis == 0):
        return ti.Vector([k, p, q])
    elif ti.static(axis == 1):
        return ti.Vector([p, k, q])
    else:
        return ti.Vector([p, q, k])


@ti.kernel
def _init_from_mask(m: ti.types.ndarray(dtype=ti.u8, ndim=3), d: ti.types.ndarray(dtype=ti.i32, ndim=3)):
    for i, j, k in d:
        d[i, j, k] = 0 if m[i, j, k] != 0 else INF


@ti.kernel
def _edt_axis(d: ti.types.ndarray(dtype=ti.i32, ndim=3),
              s: ti.types.ndarray(dtype=ti.i32, ndim=2),
              t: ti.types.ndarray(dtype=ti.i32, ndim=2),
              g: ti.types.ndarray(dtype=ti.i32, ndim=2),
              axis: ti.template(), off: ti.i32, cnt: ti.i32, nq: ti.i32, L: ti.i32):
    for b in range(cnt):
        line = off + b
        p = line // nq
        qq = line % nq
        for k in range(L):
            v = _voxel(axis, p, qq, k)
            _wr(g, b, k, d[v[0], v[1], v[2]])
        q = 0
        _wr(s, b, 0, 0)
        _wr(t, b, 0, 0)
        for u in range(1, L):
            gu = _rd(g, b, u)
            # pop parabolas hidden by the one at u; the max() guards the non-short-circuit read at q = -1
            go = True
            while go:
                go = False
                if q >= 0:
                    sq = _rd(s, b, q)
                    tq = _rd(t, b, q)
                    fa = (tq - sq) * (tq - sq) + _rd(g, b, sq)
                    fb = (tq - u) * (tq - u) + gu
                    if fa > fb:
                        q -= 1
                        go = True
            if q < 0:
                q = 0
                _wr(s, b, 0, u)
                _wr(t, b, 0, 0)
            else:
                si = _rd(s, b, q)
                w = 1 + (u * u - si * si + gu - _rd(g, b, si)) // (2 * (u - si))
                if w < L:
                    q += 1
                    _wr(s, b, q, u)
                    _wr(t, b, q, w)
        for uu in range(L):
            u = L - 1 - uu
            sq = _rd(s, b, q)
            v = _voxel(axis, p, qq, u)
            d[v[0], v[1], v[2]] = (u - sq) * (u - sq) + _rd(g, b, sq)
            if u == _rd(t, b, q):
                q -= 1


def _edt2(m, d):
    """Fill the int32 ndarray d with the exact squared distance of every voxel to the nearest voxel
    where the uint8 ndarray m is non-zero (INF where m is empty)."""
    shape = m.shape
    L_max = max(shape)
    _init_from_mask(m, d)
    B_cap = max(1, SCRATCH_BYTES // (3 * 4 * L_max))
    for axis in (2, 1, 0):
        L = shape[axis]
        others = [shape[a] for a in range(3) if a != axis]
        nq = others[1]
        nlines = others[0] * others[1]
        B = min(B_cap, nlines)
        sh = (B, L) if _cpu_layout else (L, B)
        s = ti.ndarray(ti.i32, sh)
        t = ti.ndarray(ti.i32, sh)
        g = ti.ndarray(ti.i32, sh)
        for off in range(0, nlines, B):
            _edt_axis(d, s, t, g, axis, off, min(B, nlines - off), nq, L)


def edt_sq(binary):
    """Squared distance, int32 array, of every voxel to the nearest True voxel of `binary`.
    Voxels of an all-False input get INF. Call init() first."""
    if _arch is None:
        raise RuntimeError("soft_edge_taichi.init() must be called before edt_sq()")
    m = ti.ndarray(ti.u8, binary.shape)
    m.from_numpy(np.ascontiguousarray(binary, dtype=np.uint8))
    d = ti.ndarray(ti.i32, binary.shape)
    _edt2(m, d)
    return d.to_numpy()
