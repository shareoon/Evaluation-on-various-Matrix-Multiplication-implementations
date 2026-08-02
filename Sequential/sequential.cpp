#include <iostream>
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main() {
    int N;
    cout << "Enter N: ";
    cin >> N;

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

    //Start time
    auto start = high_resolution_clock::now();

    // Matrix multiplication
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    auto stop = high_resolution_clock::now();
    //End time

    duration<double, milli> elapsed = stop-start;
    cout << elapsed.count();
    cout <<"ms\n";

    return 0;
}
