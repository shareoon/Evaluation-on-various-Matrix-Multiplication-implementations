#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;

void printMatrix(const vector<vector<int>>& M, const string& name) { 
    cout << name << ":\n"; 
    for (const auto& row : M) { 
        for (int val : row) 
            cout << val << "\t"; 
        cout << '\n'; 
    } 
    cout << '\n'; 
}

int main() {
    int t;
    int N;
    cout << "Enter N,t\n";
    cin>>N;
    cin>>t;
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

    // Matrix multiplication
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    auto stop = high_resolution_clock::now();
    //END TIME


    if(N<8){
        printMatrix(A, "Matrix A"); 
        printMatrix(B, "Matrix B"); 
        printMatrix(C, "Matrix C = A x B");        
    }
    duration<double, milli> elapsed = stop-start;
    cout << elapsed.count();
    cout <<"ms\n";

    return 0;
}