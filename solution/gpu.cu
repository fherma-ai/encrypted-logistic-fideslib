// The logistic circuit on the card, with FIDESlib's OpenFHE interop layer.
//
// The envelope makes the context, the keys and the ciphertexts with OpenFHE
// before the clock starts; this file takes them as they are. Once per point
// (gpu_init, not measured) the device context is adapted from the envelope's
// own and the relinearisation key is moved over. Per case (gpu_logistic,
// which is what the clock sees) the ciphertext goes to the card, the circuit
// runs there, and the answer comes back as an OpenFHE ciphertext the
// envelope can decrypt. Moving the data is inside the measurement on
// purpose: it is part of what answering on a GPU costs.
//
// The circuit is polycircuit's LogisticFunction, operation for operation:
//
//   σ(x) ≈ Σ c_k T_k(x/25)  over k ≤ 59, by a Paterson–Stockmeyer series
//                            on [-25, 25],
//        + Σ c_k T_k(x/25)  over the odd k in 59..77, each built by hand
//                            from a table T_1..T_63 of u = x/25 through the
//                            product formula 2 T_a T_b = T_{a+b} + T_{|a−b|}.
//
// The table's first fourteen entries are written out one by one in the
// component with a particular level plan (T_3 straight from u³, T_12 and
// T_13 repeating T_10 and T_11 — as published), the rest by the recurrence
// T_i = 2 T_j T_{i−j} − T_{|2j−i|} with j = ⌊i/2⌋. The port keeps every
// step, so the two rows on the board differ by the machine and the library.
#include "gpu.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "openfhe.h"
#ifdef duration
#undef duration  // OpenFHE defines the word; CUDA's headers use it.
#endif
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

namespace {

using GpuCiphertext = FIDESlib::CKKS::Ciphertext;

struct Device {
    FIDESlib::CKKS::Context gpu;
};

// The Chebyshev coefficients of the sigmoid on [-25, 25], degrees 0..59.
std::vector<double> SERIES = {
    1,   0.6349347497444793,      0.0, -0.207226910968973,      0.0, 0.11926554318627501,
    0.0, -0.08013715047239724,    0.0, 0.05757992161586102,     0.0, -0.04280910730211544,
    0.0, 0.03243169753850008,     0.0, -0.024837099847818355,   0.0, 0.01914266900321147,
    0.0, -0.014810271063030017,   0.0, 0.011484876770357881,    0.0, -0.008918658212582378,
    0.0, 0.006931783590193362,    0.0, -0.00539036659534493,    0.0, 0.004193061697765822,
    0.0, -0.0032623449132251907,  0.0, 0.0025385228671728505,   0.0, -0.0019754432169236547,
    0.0, 0.001537332438236821,    0.0, -0.00119641841400218,    0.0, 0.0009311199764515702,
    0.0, -0.0007246568620069462,  0.0, 0.0005639769377601768,   0.0, -0.00043892561744415164,
    0.0, 0.00034160150542931166,  0.0, -0.00026585589670811906, 0.0, 0.00020690371892740422,
    0.0, -0.00016102095964053078, 0.0, 0.0001253092833997599,
};

// The coefficients of T_59, T_61, …, T_77 (every other entry is a zero even
// degree), taken by the tail terms below in the component's own way.
const double REST[] = {
    -9.751288354472837e-05, 0.0, 7.587595780529826e-05,  0.0, -5.903178548385316e-05,  0.0,
    4.591638974071583e-05,  0.0, -3.570132724330664e-05, 0.0, 2.7741357204113592e-05,  0.0,
    -2.15336836799884e-05,  0.0, 1.668619567482025e-05,  0.0, -1.2892699205061372e-05, 0.0,
    9.91357617352584e-06,   0.0, -7.560648875376369e-06,
};

const double U = 0.04;  // x/25: the series' argument on [-1, 1]

// Fresh ciphertexts from the three-operand forms, without touching the operands.
GpuCiphertext product(FIDESlib::CKKS::Context& gpu, const GpuCiphertext& a, const GpuCiphertext& b) {
    GpuCiphertext out(gpu);
    out.mult(a, b);
    return out;
}
GpuCiphertext scaled(FIDESlib::CKKS::Context& gpu, const GpuCiphertext& a, double c) {
    GpuCiphertext out(gpu);
    out.multScalar(a, c);
    return out;
}
GpuCiphertext squared(FIDESlib::CKKS::Context& gpu, const GpuCiphertext& a) {
    GpuCiphertext out(gpu);
    out.square(a);
    return out;
}
// 2ab − 1 and 2ab − c: one recurrence step of the table.
GpuCiphertext twice_less_one(FIDESlib::CKKS::Context& gpu, const GpuCiphertext& a, const GpuCiphertext& b) {
    GpuCiphertext prod = product(gpu, a, b);
    GpuCiphertext out(gpu);
    out.add(prod, prod);
    out.addScalar(-1.0);
    return out;
}
GpuCiphertext twice_less(FIDESlib::CKKS::Context& gpu, const GpuCiphertext& a, const GpuCiphertext& b,
                         const GpuCiphertext& c) {
    GpuCiphertext prod = product(gpu, a, b);
    GpuCiphertext out(gpu);
    out.add(prod, prod);
    out.sub(c);
    return out;
}

// The table T_1..T_63 of u, as the component fills it: fourteen entries by
// hand, the rest by the recurrence. t[0] holds x itself, as t1[0] does there.
std::vector<GpuCiphertext> table(FIDESlib::CKKS::Context& gpu, const GpuCiphertext& x) {
    std::vector<GpuCiphertext> t;
    t.reserve(64);
    t.emplace_back(gpu);
    t[0].copy(x);
    t.push_back(scaled(gpu, x, U));                                   // T_1 = u
    t.push_back(twice_less_one(gpu, t[1], t[1]));                     // T_2
    {                                                                 // T_3 = 4u³ − 3u, from x
        GpuCiphertext cubic = product(gpu, scaled(gpu, x, 4 * std::pow(U, 3)), squared(gpu, x));
        GpuCiphertext three = scaled(gpu, t[1], 3.0);
        GpuCiphertext t3(gpu);
        t3.copy(cubic);
        t3.sub(three);
        t.push_back(std::move(t3));
    }
    t.push_back(twice_less_one(gpu, t[2], t[2]));                     // T_4
    t.push_back(twice_less(gpu, t[2], t[3], t[1]));                   // T_5
    t.push_back(twice_less_one(gpu, t[3], t[3]));                     // T_6
    t.push_back(twice_less(gpu, t[3], t[4], t[1]));                   // T_7
    t.push_back(twice_less_one(gpu, t[4], t[4]));                     // T_8
    t.push_back(twice_less(gpu, t[4], t[5], t[1]));                   // T_9
    t.push_back(twice_less_one(gpu, t[5], t[5]));                     // T_10
    t.push_back(twice_less(gpu, t[5], t[6], t[1]));                   // T_11
    t.push_back(twice_less_one(gpu, t[5], t[5]));                     // T_12, as published
    t.push_back(twice_less(gpu, t[5], t[6], t[1]));                   // T_13, as published
    t.push_back(twice_less_one(gpu, t[7], t[7]));                     // T_14
    for (int i = 15; i <= 63; ++i) {
        const int j = i / 2;
        if (2 * j == i) t.push_back(twice_less_one(gpu, t[j], t[i - j]));
        else t.push_back(twice_less(gpu, t[j], t[i - j], t[1]));
    }
    return t;
}

// acc ← acc · t[q]  −  c·s · t[other], one rung of a tail term.
void rung(FIDESlib::CKKS::Context& gpu, GpuCiphertext& acc, const GpuCiphertext& power,
          const GpuCiphertext& other, double cs) {
    acc.mult(power);
    GpuCiphertext away = scaled(gpu, other, cs);
    acc.sub(away);
}

}  // namespace

void* gpu_init(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
    // The device context, adapted from the envelope's: its primes, its
    // digits, its scaling technique. FIDESlib's own parameters only add the
    // batch it processes limbs in.
    FIDESlib::CKKS::RawParams raw = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Parameters seed{};
    seed.batch = 100;
    FIDESlib::CKKS::Context gpu = FIDESlib::CKKS::GenCryptoContextGPU(seed.adaptTo(raw), {0});

    // The relinearisation key, read out of the context's own key store. The
    // envelope made one key pair, so the store holds one entry; the key is
    // the first of its vector, as it is in OpenFHE's own EvalMult.
    auto& store = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetAllEvalMultKeys();
    if (store.empty()) throw std::runtime_error("gpu_init: the context holds no evaluation key");
    auto relin = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
        store.begin()->second.at(0));
    if (!relin) throw std::runtime_error("gpu_init: the evaluation key is not a relinearisation key");

    FIDESlib::CKKS::RawKeySwitchKey rawKey = FIDESlib::CKKS::GetKeySwitchKey(relin);
    FIDESlib::CKKS::KeySwitchingKey key(gpu);
    key.Initialize(rawKey);
    gpu->AddEvalKey(std::move(key));

    return new Device{std::move(gpu)};
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_logistic(
    void* state,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct) {
    Device& device = *static_cast<Device*>(state);
    FIDESlib::CKKS::Context& gpu = device.gpu;

    // To the card.
    FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, ct);
    GpuCiphertext x(gpu, raw);

    // The series over degrees ≤ 59, on [-25, 25].
    GpuCiphertext out(gpu);
    out.copy(x);
    FIDESlib::CKKS::evalChebyshevSeries(out, SERIES, -25.0, 25.0);

    // The table, and the odd terms above it — each spelled out as the
    // component spells it, rung by rung.
    std::vector<GpuCiphertext> t = table(gpu, x);
    GpuCiphertext sum(gpu);

    {  // T_59
        const double c = REST[0];
        GpuCiphertext acc = product(gpu, scaled(gpu, t[3], c * 8), t[8]);
        acc.sub(scaled(gpu, t[5], c * 4));
        rung(gpu, acc, t[16], t[5], c * 2);
        rung(gpu, acc, t[32], t[5], c);
        sum.copy(acc);
    }
    {  // T_61
        const double c = REST[2];
        GpuCiphertext acc = product(gpu, scaled(gpu, t[1], c * 16), t[4]);
        acc.sub(scaled(gpu, t[3], c * 8));
        rung(gpu, acc, t[8], t[3], c * 4);
        rung(gpu, acc, t[16], t[3], c * 2);
        rung(gpu, acc, t[32], t[3], c);
        sum.add(acc);
    }
    {  // T_63, with T_65's coefficient taken off it here and put back below
        const double c = REST[4] - REST[6];
        GpuCiphertext acc = product(gpu, scaled(gpu, t[1], c * 32), t[2]);
        acc.sub(scaled(gpu, t[1], c * 16));
        rung(gpu, acc, t[4], t[1], c * 8);
        rung(gpu, acc, t[8], t[1], c * 4);
        rung(gpu, acc, t[16], t[1], c * 2);
        rung(gpu, acc, t[32], t[1], c);
        sum.add(acc);
    }
    {  // T_65: its cube straight from x
        const double c = REST[6];
        GpuCiphertext acc = product(gpu, scaled(gpu, x, (c * 32 * 4) / std::pow(25.0, 3)), squared(gpu, x));
        acc.sub(scaled(gpu, t[1], c * 64));
        rung(gpu, acc, t[2], t[1], c * 32);
        rung(gpu, acc, t[4], t[1], c * 16);
        rung(gpu, acc, t[8], t[1], c * 8);
        rung(gpu, acc, t[16], t[1], c * 4);
        rung(gpu, acc, t[32], t[1], c * 2);
        sum.add(acc);
    }
    for (int i = 0; i < 6; ++i) {  // T_67 … T_77
        const double c = REST[8 + 2 * i];
        GpuCiphertext cubic = product(gpu, scaled(gpu, x, 32 * c * 4 * std::pow(U, 3)), squared(gpu, x));
        GpuCiphertext acc(gpu);
        acc.copy(cubic);
        acc.sub(scaled(gpu, x, 3 * 32 * c * U));
        rung(gpu, acc, t[2], t[1], c * 16);
        rung(gpu, acc, t[4], t[1], c * 8);
        rung(gpu, acc, t[8], t[1], c * 4);
        rung(gpu, acc, t[16], t[1], c * 2);
        rung(gpu, acc, t[34 + 2 * i], t[1 + 2 * i], c);
        sum.add(acc);
    }
    out.add(sum);

    // Back, as an OpenFHE ciphertext the envelope can decrypt.
    FIDESlib::CKKS::RawCipherText rawOut;
    out.store(rawOut);
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> answer = ct->Clone();
    FIDESlib::CKKS::GetOpenFHECipherText(answer, rawOut);
    return answer;
}

void gpu_free(void* state) { delete static_cast<Device*>(state); }
