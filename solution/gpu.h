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

// The logistic function over one ciphertext: to the card, the circuit, back. Measured.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_logistic(
    void* state,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct);

void gpu_free(void* state);
