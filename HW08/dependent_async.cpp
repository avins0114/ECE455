#include <taskflow/taskflow.hpp>

int main() {
    tf::Executor executor;
    tf::AsyncTask A = executor.async([]() {
        printf("A\n");
    });
    tf::AsyncTask B = executor.async([]() {
        printf("B\n");
    }, A);
    tf::AsyncTask C = executor.async([]() {
        printf("C\n");
    }, A);

    auto [D, fuD] = executor.dependent_async([]() {
        printf("D\n");
    }, B, C);

    fuD.get(); // wait for D to finish
}