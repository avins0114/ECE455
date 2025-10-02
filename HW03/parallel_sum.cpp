#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    const int N = 1'000'000;

    std::vector<int> data(N, 1); // make arr of all 1's
    long long total = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        total += data[i];
    }

    std::cout << "Sum = " << total << std::endl;
    return 0;

}