// GENERATED. The four functions that are yours, declared once.
#pragma once

#include "fherma.h"
#include "openfhe.h"

using lbcrypto::Ciphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;
using lbcrypto::Plaintext;

// Setup over public material. Once per point, not measured.
void* solve_init(const fherma::Point& p, CryptoContext<DCRTPoly> cc);

// The layout: cleartext into plaintext packings, one per ciphertext.
// Reviewed together with solve_decoding as one technique.
std::vector<Plaintext> solve_encoding(CryptoContext<DCRTPoly> cc,
                                      const fherma::Inputs& inp);

// The answer, over ciphertexts. Measured — and only this.
std::vector<Ciphertext<DCRTPoly>> solve_run(
    void* state,
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& cts);

// Reading the answer out of the decrypted packings. Not measured.
fherma::Outputs solve_decoding(const fherma::Point& p,
                               CryptoContext<DCRTPoly> cc,
                               const std::vector<Plaintext>& pts);

void solve_free(void* state);

// Optional, for an answer that keeps its ciphertexts on a device: host to
// device before solve_run, device to host after it. Each is clocked on its
// own and neither is the score; left out, neither is called.
void fherma_h2d(
    void* state,
    CryptoContext<DCRTPoly>,
    const std::vector<Ciphertext<DCRTPoly>>& cts);
std::vector<Ciphertext<DCRTPoly>> fherma_d2h(
    void* state,
    CryptoContext<DCRTPoly>,
    std::vector<Ciphertext<DCRTPoly>> out);

// Optional: numbers only the answer knows about itself, by name — kernels
// launched, bytes held on a device — read after every case and reported in
// its row's metrics. Never the score.
std::vector<std::pair<std::string, double>> fherma_metrics(void* state);
