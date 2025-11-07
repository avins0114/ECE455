#include <taskflow/taskflow.hpp>

int main() {
    tf::Executor executor;;
    tf::Taskflow taskflow("static taskflow demo");

    auto A = taskflow.emplace([]() {
        printf("Task A is running\n");
    });
    auto B = taskflow.emplace([]() {
        printf("Task B is running\n");
    });
    auto C = taskflow.emplace([]() {
        printf("Task C is running\n");
    }); 
    auto D = taskflow.emplace([]() {
        printf("Task D is running\n");
    });

    A.precede(B, C); // A must finish before B and C start
    B.precede(D);     // B must finish before D starts
    C.precede(D);     // C must finish before D starts

    executor.run(taskflow).wait();
}