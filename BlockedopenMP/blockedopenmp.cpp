#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;

int main() {
    int N, t;
    int BS=48;
    cout << "Enter N: \n";
    cin >> N;
    cout << "Enter thread no: \n";
    cin >> t; 
    omp_set_num_threads(t);

    srand(42);

    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N));

    // Generate random matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    //START TIME
    auto start = high_resolution_clock::now();

    // Blocked Matrix multiplication
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < N; ii += BS) {
        for (int jj = 0; jj < N; jj += BS) {
            for (int kk = 0; kk < N; kk += BS) {

                // Multiply one block
                for (int i = ii; i < min(ii + BS, N); i++) {
                    for (int j = jj; j < min(jj + BS, N); j++) {
                        for (int k = kk; k < min(kk + BS, N); k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }

            }
        }
    }

    auto stop = high_resolution_clock::now();
    //END TIME

    duration<double, milli> elapsed = stop-start;
    cout << elapsed.count();
    cout <<"ms\n";

    return 0;
}