#  — an encrypted answer on a GPU

**This is `solution` with the copies to and from the card outside the
measured call.** `fherma_h2d` loads the case onto the card (`gpu_h2d`),
`solve_run` is the evaluation and a device synchronize (`gpu_logistic`), and
`fherma_d2h` brings the answer back as an OpenFHE ciphertext (`gpu_d2h`). The
loop reports the two copies as `h2d_s` and `d2h_s` beside the score
(fherma-lang 0.20.0 and later). The arithmetic is `solution`'s, call for call.


Answers `logistic` over ciphertexts, with FIDESlib (CKKS) under OpenFHE's envelope.

Yours to write: **gpu.cu** — the circuit, on the card — and the packing in
**solve.cpp**. Everything else is generated from the signature and the config,
and `--update` rewrites it without touching yours.

| | |
| --- | --- |
| secret arguments | xs — arrive in `run` as OpenFHE ciphertexts |
| public arguments | none — arrive as plain values |
| measured | the `run` call alone: the copy to the card, the circuit, the copy back |
| not measured | `gpu_init`: the device context and the keys, once per point |
| keys | generated and held by the envelope; `gpu_init` reads the evaluation keys out of the context's store |
| parameters | `config.jsonc`: the OpenFHE parameters; the device context is adapted from them |
| builds in | `fherma/fideslib:2.1.3` — FIDESlib 2.1.3 with a patched OpenFHE 1.5.1, CUDA 12.9, compiled for sm_89 and sm_120 |

The context is OpenFHE's, so `config.jsonc` reads exactly as it does for an
OpenFHE answer, and the security level in it is enforced by the library at
context creation. FIDESlib answers CKKS only. The first compile happens on a
machine with a card: the image has no CPU fallback.
