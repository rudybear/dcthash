# DCTHash format, version 0

DCTHash is a compact text representation of an image placeholder: 6 to 96 characters that decode
to a blurry preview, an average colour and an approximate aspect ratio. It is the production
form of the `dct_fixed` codec from this repository's experiment (see `README.md`).



This document is normative together with `format/tables.json`, which holds every table as
integers. `format/vectors.json` holds test vectors. The reference implementations are
`hash2d/dcthash.py` (Python) and `ts/` (TypeScript).

- SHA-256 of `format/tables.json` for this version:
  `1c05ee1b6fca2fb87e9edef3a462b09b8074e411dbe156bd205c1277b9413081`

Conformance:

- **Encoders** are fully specified in integer arithmetic. Two conforming encoders given the
  same pixels, size and length MUST produce the identical string.
- **Decoders** are specified up to floating-point rounding. Decoded 8-bit channel values MUST be
  within 1 of the reference. (The two reference decoders agree exactly on all vectors.)
- Every string that passes the validity rules in section 2 decodes; there are no further
  error conditions.

All divisions written `floor(a / b)` are mathematical floor divisions (round toward minus
infinity). All intermediate values in the encoder are integers below 2^53, so they are exact in
IEEE doubles as well as in 64-bit integers.

## 1. Overview

```
pixels --box filter--> 32x32 grid --integer YCbCr--> 16x16 DCT-II coefficients per channel
       --> quantise under 5 candidates (photo profile x 4 contrast scales, smooth profile)
       --> keep the candidate with the smallest error --> pack bits --> base64url text
```

The decoder reads the header, de-quantises the coefficients and evaluates the cosine series
directly at the requested output size.

Two *profiles* (table sets) exist because a fixed-length code cannot adapt its quantiser ranges
per image:

- **photo**: coefficient order, means, scales and bit allocation trained on photographs. Many
  coefficients, 1 to 8 bits each, uniform quantisers with ranges of a few standard deviations.
- **smooth**: for gradients, flat colours and other content whose coefficients lie far outside
  photo statistics. Fewer coefficients, each with a full-range companded quantiser that has an
  exact zero level.

The encoder picks per image; one header bit records the choice.

## 2. Text encoding and validity

Alphabet (index 0 to 63), the RFC 4648 base64url alphabet without padding:

```
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_
```

Each character carries 6 bits, most significant bit first; the string is the bit stream in
order. The characters are safe unescaped in URLs (path, query and fragment), JSON, HTML
attributes, CSV and SQL string literals.

A string is a valid DCTHash v0 if and only if:

1. its length N satisfies 6 <= N <= 96;
2. every character is in the alphabet;
3. the first bit (the top bit of the first character) is 0. This is the version bit; a set bit
   is reserved for future versions and MUST be rejected.

Any length in 6..96 is allowed. This is possible because the bit allocation is a single nested
sequence (section 5): a longer hash refines a shorter one's allocation, so no per-length tables
exist. Recommended lengths: 28 (the length of a default 4x3 BlurHash), 20 for lists of many
items, 40 to 60 for large hero images. Hashes of different lengths are not prefixes of each
other.

## 3. Bit stream layout

| bits | field | meaning |
|---|---|---|
| 1 | version | 0 |
| 5 | aspect | index into the aspect table (section 4); 31 = unspecified |
| 1 | profile | 0 = photo, 1 = smooth |
| 2 | scale | photo profile only: contrast scale code 0..3 (section 6) |
| rest | payload | quantiser indices, one field per coded coefficient, in table order, MSB first |

The payload has exactly `P = 6N - 9` bits (photo) or `P = 6N - 7` bits (smooth) and is always
completely used.

## 4. Aspect ratio

Aspect codes 0..30 denote these width:height ratios, in increasing order
(`tables.json: aspects`):

```
1:5 1:4 1:3 5:12 3:7 4:9 1:2 9:16 3:5 5:8 2:3 5:7 3:4 4:5 5:6 1:1
6:5 5:4 4:3 7:5 3:2 8:5 5:3 16:9 2:1 9:4 7:3 12:5 3:1 4:1 5:1
```

Code 31 means "unspecified". The aspect field is informational: it does not affect decoding of
the picture. It lets a client reserve layout space from the string alone.

Encoder rule (nearest ratio on a logarithmic scale, in integers): with `p_k:q_k` the k-th ratio,

```
aspect = number of k in 0..29 such that  w*w * q_k * q_(k+1)  >=  h*h * p_k * p_(k+1)
```

An encoder MAY instead write 31.

## 5. Tables and bit allocation

`tables.json` contains, per profile, a list of coefficients `coefs[i] = [channel, u, v]`
(channel 0 = Y, 1 = Cb, 2 = Cr; u = horizontal frequency, v = vertical frequency, both 0..15)
and a bit sequence `seq`. The photo profile also has a centre `M[i]` and a scale `S[i]` per
coefficient, and there is one global list `g[1..8]` of loading factors (times 256).

| | photo | smooth |
|---|---|---|
| coefficients | 213 (u, v up to 15) | 86 (u up to 7, v up to 6) |
| sequence length | 569 | 569 |

**Bit allocation.** The number of bits `b_i` of coefficient i in a hash with P payload bits is
the number of occurrences of i among the first P entries of `seq`. No coefficient ever gets
more than 8 bits. Coefficients with `b_i = 0` are not transmitted.

Units: coefficient values are expressed in **T units**. With `A(c,u,v) = sum over the 32x32
grid of p(x,y) * cos(pi u (2x+1)/64) * cos(pi v (2y+1)/64)` (p in 8-bit levels, no
normalisation), `T = A / 4`. The constant `FULL = 32640` is 127.5 levels for a DC term, half
of the 8-bit range.

## 6. Quantisers

Each coefficient uses one of two quantisers.

| coefficient | photo profile | smooth profile |
|---|---|---|
| luma DC (channel 0, u = v = 0) | uniform, M = FULL, L = FULL | uniform, M = FULL, L = FULL |
| chroma DC (channel 1 or 2, u = v = 0) | companded | companded |
| everything else | uniform, M = M[i], L from S[i] | companded |

**Uniform quantiser** with centre M, half-range L and b bits. Index q in 0..2^b - 1.

```
reconstruction:  value = M - L + (2q + 1) * L / 2^b
```

For photo-profile coefficients other than DC, with scale code s and `SCALES = [2, 3, 4, 6]`:

```
L0 = max(1, floor((S[i] * g[b] + 128) / 256))
L  = max(1, floor(L0 * SCALES[s] / 4))
```

So the four scale codes multiply the trained ranges by 0.5, 0.75, 1 and 1.5.

**Companded quantiser** with b bits. Index q in 0..2^b - 1 maps to a signed step
`n = q - (2^(b-1) - 1)`, so n runs from `-(2^(b-1) - 1)` to `2^(b-1)` and n = 0 is exactly zero.

```
reconstruction:  value = FULL * n * |n| / 4^(b-1)
```

The largest magnitude is FULL, which bounds every possible coefficient.

A coefficient with `b = 0` reconstructs to its centre: M[i] in the photo profile; in the
smooth profile FULL for luma DC and 0 otherwise. Coefficients that are not in the profile's
table are 0.

Luma DC is uniform over the whole 8-bit range and chroma DC is companded over the whole gamut
so that solid black, white and saturated images are representable at every length.

## 7. Decoding

1. Validate (section 2), read the header, select the profile, compute the allocation and read
   the indices.
2. De-quantise to `value_i` in T units and scale:
   `K(c,u,v) = value * a(u) * a(v) / 256`, with `a(0) = 1` and `a(f) = 2` for f > 0.
3. For an output of W x H pixels, for each channel c:

   ```
   plane_c(x, y) = sum over (u, v) of K(c,u,v) * cos(pi u (x + 0.5) / W) * cos(pi v (y + 0.5) / H)
   ```

4. Convert to RGB and round:

   ```
   R = Y + 1.402 Cr
   G = Y - 0.344136 Cb - 0.714136 Cr
   B = Y + 1.772 Cb
   byte = clamp(floor(value + 0.5), 0, 255)
   ```

   Alpha is 255.

The picture is defined on the unit square, so any W x H can be requested; the placeholder is
stretched to it. Decoding at a small size (for example 32 px on the long side) and letting the
display scale it up is the intended fast path.

**Average colour** without decoding pixels: the three DC values, `Y = K(0,0,0)`,
`Cb = K(1,0,0)`, `Cr = K(2,0,0)`, converted as in step 4. This is the exact mean of the
unclamped picture; clamping at 0/255 can move the mean of the decoded bytes away from it.

The reference decoders limit W and H to 1..4096 and W*H to 2^22; this is an implementation
limit, not part of the format.

## 8. Encoding (normative, integer only)

Input: W x H pixels, 8-bit RGBA with straight alpha, 1 <= W, H <= 16384; a length N.

1. **Alpha.** Composite each pixel over white:
   `c' = floor((c * a + 255 * (255 - a) + 127) / 255)` for c in R, G, B.
2. **Box filter to 32x32.** Source column i covers `[32i, 32i + 32)` and grid column j covers
   `[W j, W j + W)` on a common axis of length 32W; `ox(i, j)` is the length of their overlap
   (an integer). Likewise `oy` with H. For each grid cell and colour:

   ```
   sum(j, k) = sum over pixels (i, r) of ox(i, j) * oy(r, k) * c'(i, r)
   V(j, k)   = floor((sum * 16 + floor(W*H / 2)) / (W*H))          (units of 1/16 level, 0..4080)
   ```

   This is exact area averaging for any size, including images smaller than 32 px.
3. **Colour.**

   ```
   Y  = floor(( 19595 R + 38470 G +  7471 B + 32768) / 65536)
   Cb = floor((-11059 R - 21709 G + 32768 B + 32768) / 65536)
   Cr = floor(( 32768 R - 27439 G -  5329 B + 32768) / 65536)
   ```

4. **DCT.** With the frozen table `COS[f][x] = round(2^14 cos(pi f (2x + 1) / 64))`
   (`tables.json: cos`, f in 0..15, x in 0..31), for every channel and u, v in 0..15:

   ```
   c(u, v) = sum over x, y of COS[u][x] * COS[v][y] * P(x, y)
   X(u, v) = floor((c + 2^27) / 2^28)
   ```

   X equals 16 A, that is 64 T units. (No table entry lies within 0.01 of a rounding tie, so
   computing the table with any reasonable `cos` gives the same integers.)
5. **Quantise** under each of five candidates, in this order: photo with scale code 0, 1, 2, 3;
   then smooth. For each coded coefficient:

   Uniform: `q = clamp(floor((X - 64 M + 64 L) * 2^b / (128 L)), 0, 2^b - 1)`.

   Companded: with `m_k = n_k * |n_k|` and `n_k = k - (2^(b-1) - 1)`,
   `q = number of k in 1..2^b - 1 such that  2 X 4^(b-1) >= 64 FULL (m_(k-1) + m_k)`
   (the nearest level; a value exactly at a midpoint takes the higher level).
6. **Choose the candidate.** Let `R(c,u,v)` be the reconstruction times 2^16:

   ```
   uniform:    R = (M - L) * 2^16 + (2q + 1) * L * 2^(16 - b)
   companded:  R = FULL * n * |n| * 2^(18 - 2b)
   b = 0:      R = centre * 2^16          not in the profile's table:  R = 0
   ```

   Over all 3 x 16 x 16 coefficients, with `CW = [12, 13, 10]` for Y, Cb, Cr:

   ```
   e = floor((X * 1024 - R) / 32768)
   E = sum of CW[c] * a(u) * a(v) * e * e
   ```

   Pick the candidate with the smallest E; on a tie, the earliest in the order of step 5.
7. **Pack** the header and indices (section 3) and map each 6 bits to a character.

## 9. Design notes: what changed from the experiment's `dct_fixed`

| change | reason | effect on Kodak mean PSNR at 28 chars |
|---|---|---|
| base64url (6 bits per char) instead of bytes or base83 | URL/JSON/DB safe, no big-integer arithmetic | fewer bits than BlurHash at equal length (6 vs 6.375 per char) |
| any length 6..96 through one nested bit sequence | no per-length tables | none |
| aspect ratio stored in the string (5 bits, 31 exact common ratios) | decoder no longer needs the size to choose tables; layout hint | costs 5 bits |
| one table set for all aspect ratios | 3x smaller tables | -0.00 to -0.02 dB |
| quantiser range = S[i] x g[b] instead of a free table per (i, b) | 8 numbers instead of 1700 | within 0.01 dB |
| cosines evaluated at the output size instead of 32x32 + bicubic | smaller decoder, exact at any size | none measurable |
| luma DC uniform over 0..255; chroma DC companded over the full gamut | solid white decoded as 245 and pure red as (143, 59, 3) before | none |
| 2-bit contrast scale | adapts ranges to low/high-contrast photos | +0.08 dB |
| smooth profile | photo tables gave 17.7 dB on a two-colour gradient where BlurHash gives 31.9 dB, and textured noise on flat images | none on Kodak (never selected at 20+ chars); gradients now 34 dB |
| integer-only encoder | identical strings across implementations | none |

The net result on Kodak at equal total bits is the same as the experiment's `dct_fixed`
(18.13 dB at 28 chars = 168 bits, versus 18.11 dB for `dct_fixed` at 21 bytes = 168 bits).

## 10. Known limits

- No alpha channel; transparent pixels are composited over white.
- Hard-edged flat graphics (logos, UI screenshots) are reproduced poorly by any 28-character
  code; DCTHash is somewhat ahead of BlurHash there (12.6 vs 10.8 dB on one test image) but
  neither is good.
- In the photo profile, chroma gets few bits (9 of 64 coded coefficients at 28 characters).
  That maximises PSNR on photographs, but strongly coloured small objects come out less
  saturated than in BlurHash.
- The profile decision minimises squared error in the coefficient domain, not a perceptual
  measure.
- The aspect ratio is one of 31 values chosen to be exact for common ratios. An unusual ratio
  is rounded to the nearest: off by up to 9.5% between 1:2 and 2:1 (worst just beside 1:1) and
  up to 15.5% between 3:1 and 4:1. A uniform log scale in the same 5 bits would be within 4.4%
  everywhere but exact for none of 3:2, 4:3 or 16:9. Ratios beyond 1:5 or 5:1 saturate.
- One spare version bit. A second version would have to fit behind it.
