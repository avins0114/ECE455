#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>

void multiply_block(const std::vector<double> &A,
                    const std::vector<double> &B,
                    std::vector<double> &C,
                    int N, int row_start, int row_end) {
    for (int i = row_start; i < row_end; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int N = 800;
    const int T = 4;  // Fixed number of threads like other problems
    
    std::vector<double> A(N * N), B(N * N), C(N * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (auto &x : A) x = dist(rng);
    for (auto &x : B) x = dist(rng);

    // Single-threaded baseline
    std::vector<double> C_single(N * N);
    auto t0 = std::chrono::high_resolution_clock::now();
    multiply_block(A, B, C_single, N, 0, N);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Parallel version
    std::vector<std::thread> threads;
    int chunk = N / T;
    auto p0 = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < T; ++t) {
        int rs = t * chunk;
        int re = (t == T - 1) ? N : rs + chunk;
        threads.emplace_back(multiply_block,
                             std::cref(A), std::cref(B), std::ref(C),
                             N, rs, re);
    }
    
    for (auto &th : threads) th.join();
    auto p1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> t_single = t1 - t0;
    std::chrono::duration<double> t_par = p1 - p0;
    
    std::cout << "Single-threaded multiplication took " << t_single.count() << " s\n";
    std::cout << "Parallel multiplication took " << t_par.count() << " s\n";
    std::cout << "Speedup: " << (t_single.count() / t_par.count()) << "x\n";
    
    return 0;
}
