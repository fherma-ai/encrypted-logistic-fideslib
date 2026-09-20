// The logistic function of every element, over CKKS, computed on the GPU
// with FIDESlib.
//
// A port. The circuit is the one the CPU answer runs — the LogisticFunction
// component from fairmath/polycircuit (Apache-2.0), the winning entry of the
// FHERMA Logistic Function challenge by Aikata (TU Graz): a Chebyshev series
// of the sigmoid on [-25, 25] to T₅₉, with the odd terms up to T₇₇ unrolled
// by hand through the product formula, in seven levels. Moved operation for
// operation onto FIDESlib's device ciphertexts; the mathematics is hers and
// untouched.
//
// This file is the plain C++ half: how the vector is packed and read back.
// The circuit itself is in gpu.cu, which is compiled by nvcc.
#include "solve.h"

#include "gpu.h"

void* solve_init(const fherma::Point& p, CryptoContext<DCRTPoly> cc) {
    // The device context and the relinearisation key, before the clock.
    return gpu_init(cc);
}

std::vector<Plaintext> solve_encoding(CryptoContext<DCRTPoly> cc,
                                      const fherma::Inputs& inp) {
    // The whole vector in one packing, slot i holding element i.
    std::vector<double> xs(inp.xs.data.begin(), inp.xs.data.end());
    return { cc->MakeCKKSPackedPlaintext(xs) };
}

std::vector<Ciphertext<DCRTPoly>> solve_run(
    void* state,
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& cts) {
    return { gpu_logistic(state, cc, cts[0]) };
}

fherma::Outputs solve_decoding(const fherma::Point& p,
                               CryptoContext<DCRTPoly> cc,
                               const std::vector<Plaintext>& pts) {
    auto values = pts[0]->GetRealPackedValue();

    fherma::Outputs out;
    out.y.shape = { static_cast<int64_t>(p.N) };
    out.y.data.assign(values.begin(), values.begin() + p.N);
    return out;
}

void solve_free(void* state) { gpu_free(state); }
