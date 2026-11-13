// The device side of the answer, behind a C++ face.
//
// Everything that includes a FIDESlib header is CUDA and lives in gpu.cu, so
// solve.cpp and the platform's envelope stay plain C++. These two functions
// are the whole boundary: the device context made once per point from the
// envelope's own, and one run of the circuit over one ciphertext.
#pragma once

#include "openfhe.h"

// The GPU context adapted from `cc`, with its relinearisation key on the card.
// Not measured: called from solve_init.
void* gpu_init(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc);

// The case to the card. Clocked as h2d_s, not the score.
void gpu_h2d(
    void* state,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct);

// The evaluation on the card, until it has finished. Measured.
void gpu_logistic(void* state);

// The answer back, as an OpenFHE ciphertext. Clocked as d2h_s, not the score.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_d2h(void* state);

void gpu_free(void* state);
