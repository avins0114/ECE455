#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <cmath>

void multiply_block(const std::vector<double> &A,
                    const std::vector<double> &B,
                    std::vector<double> &C,
                    int N, int row_start, int row_end)
{
    for (int i = row_start; i < row_end; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    const int N = 800;
    const int T = std::thread::hardware_concurrency() ? 
                  std::thread::hardware_concurrency() : 4;
    
    std::vector<double> A(N * N), B(N * N), C(N * N, 0.0);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (auto &x : A)
        x = dist(rng);
    for (auto &x : B)
        x = dist(rng);

    // Single-threaded baseline
    std::vector<double> C_single(N * N, 0.0);
    auto single_start = std::chrono::high_resolution_clock::now();

    multiply_block(A, B, C_single, N, 0, N);

    auto single_end = std::chrono::high_resolution_clock::now();

    // Parallel version
    std::vector<std::thread> threads;
    int chunk = N / T;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < T; ++t)
    {
        int rs = t * chunk;
        int re = (t == T - 1) ? N : rs + chunk;
        threads.emplace_back(multiply_block,
                             std::cref(A), std::cref(B), std::ref(C),
                             N, rs, re);
    }
    
    for (auto &th : threads)
        th.join();
    
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate timing
    double single_time = std::chrono::duration<double>(single_end - single_start).count();
    double parallel_time = std::chrono::duration<double>(end_time - start_time).count();

    // Verify results match
    bool results_match = true;
    for (size_t i = 0; i < N * N; ++i) {
        if (std::abs(C[i] - C_single[i]) > 1e-10) {
            results_match = false;
            break;
        }
    }

    // Prevent compiler optimization by using results
    volatile double dummy = C_single[0] + C[N*N-1];
    (void)dummy; // Suppress unused variable warning

    // Output results
    std::cout << "Single-threaded multiplication took " << single_time << " s\n";
    std::cout << "Parallel multiplication took " << parallel_time << " s\n";
    std::cout << "Speedup: " << (single_time / parallel_time) << "x\n";
    std::cout << "Using " << T << " threads\n";
    std::cout << "Results match: " << (results_match ? "Yes" : "No") << "\n";

    return 0;
}