#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <immintrin.h> 
#include <cstdlib>

using namespace std;
using namespace std::chrono;

const int embedding_table_size = 2000000;
const int embedding_dim = 64;
const int input_size = 720;
const int num_bags = 20;
const int prefetch = 16;


int random_int(int range) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(0, range - 1);
    return dis(gen);
}

long long run_with_prefetching(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
            int start_idx = offsets[i];
            int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

            vector<float> bag_embedding(embedding_dim, 0.0f);

            for (int j = start_idx; j < end_idx; ++j) {
                float* data_ptr = (float*)&embedding_table[input[j] * embedding_dim];
                if (j+3 < end_idx) {
                    int idx_pref = input[j + 3];
                    // _mm_prefetch((const char*)&embedding_table[idx_pref * embedding_dim], _MM_HINT_T0);
                    // _mm_prefetch((const char*)&embedding_table[input[j+2] * embedding_dim], _MM_HINT_T0);
                    // _mm_prefetch((const char*)&embedding_table[input[j+3] * embedding_dim], _MM_HINT_T0);
                    // _mm_prefetch((const char*)&embedding_table[input[j+4] * embedding_dim], _MM_HINT_T0);
                    // _mm_prefetch((const char*)&embedding_table[input[j+5] * embedding_dim], _MM_HINT_T0);
                    // _mm_prefetch((const char*)&embedding_table[input[j+6] * embedding_dim], _MM_HINT_T0);
                    
                    __builtin_prefetch((const void*)&embedding_table[idx_pref * embedding_dim], 0, 0);
                }
                for (int d = 0; d < embedding_dim; ++d) {
                    // if (d%16 == 0 and d+16 < embedding_dim) {
                    //     _mm_prefetch(&data_ptr[d+16], _MM_HINT_T0);
                    // }
                    bag_embedding[d] += data_ptr[d];
                }
            }
            output.push_back(bag_embedding);
        }
    //-------------------------------------------------------------------------------------------------------------------------------------------
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching: " << duration.count() << " microseconds.";

    return duration.count();
}

long long run_with_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {
    auto start = high_resolution_clock::now();
    
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            const float* data_ptr = &embedding_table[(input[j]) * embedding_dim ];

            int d = 0;
            for (; d + 16 <= embedding_dim; d += 16) {
                __m512 acc = _mm512_loadu_ps(&bag_embedding[d]);
                __m512 val = _mm512_loadu_ps(&data_ptr[d]);
                acc = _mm512_add_ps(acc, val);
                _mm512_storeu_ps(&bag_embedding[d], acc);
            }
        }
        output.push_back(bag_embedding);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH SIMD: " << duration.count() << " microseconds.";
    return duration.count();
}

long long run_with_prefetching_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {
        auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
        vector<vector<float>> output;

        for (size_t i = 0; i < offsets.size(); ++i) {
                int start_idx = offsets[i];
                int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

                vector<float> bag_embedding(embedding_dim, 0.0f);

                for (int j = start_idx; j < end_idx; ++j) {
                    float* data_ptr = (float*)&embedding_table[input[j] * embedding_dim];
                    if (j+3 < end_idx) {
                        int idx_pref = input[j + 3];
                        _mm_prefetch((const char*)&embedding_table[idx_pref * embedding_dim], _MM_HINT_T0);
                        // _mm_prefetch((const char*)&embedding_table[input[j+2] * embedding_dim], _MM_HINT_T0);
                        // _mm_prefetch((const char*)&embedding_table[input[j+3] * embedding_dim], _MM_HINT_T0);
                        // _mm_prefetch((const char*)&embedding_table[input[j+4] * embedding_dim], _MM_HINT_T0);
                        // _mm_prefetch((const char*)&embedding_table[input[j+5] * embedding_dim], _MM_HINT_T0);
                        // _mm_prefetch((const char*)&embedding_table[input[j+6] * embedding_dim], _MM_HINT_T0);
                        
                        // __builtin_prefetch((const void*)&embedding_table[idx_pref * embedding_dim], 0, 0);
                    }
                    int d = 0;
                    for (; d + 16 <= embedding_dim; d += 16) {
                        __m512 acc = _mm512_loadu_ps(&bag_embedding[d]);
                        __m512 val = _mm512_loadu_ps(&data_ptr[d]);
                        acc = _mm512_add_ps(acc, val);
                        _mm512_storeu_ps(&bag_embedding[d], acc);
                    }
                }
                output.push_back(bag_embedding);
            }
        //-------------------------------------------------------------------------------------------------------------------------------------------
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        cout << "\nTime WITH software prefetching: " << duration.count() << " microseconds.";

        return duration.count();
}

long long naive_emb(vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {
    auto start = high_resolution_clock::now();
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) {
                bag_embedding[d] += data_ptr[d];
            }
        }
        output.push_back(bag_embedding);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITHOUT software prefetching: " << duration.count() << " microseconds.";
    return duration.count();
}

int main(int argc, char* argv[]) {
    // Prepare embedding table
    vector<float> embedding_table(embedding_table_size * embedding_dim);
    for (auto& val : embedding_table) {
        val = static_cast<float>(random_int(embedding_table_size));
    }

    // Input indices
    vector<int> input(input_size);
    for (auto& idx : input) {
        idx = random_int(embedding_table_size);
    }

    // Offsets
    vector<int> offsets;
    for (int i = 0; i < num_bags; ++i) {
        offsets.push_back((input_size * i) / num_bags);
    }

    string mode = (argc > 1) ? string(argv[1]) : "both";

    if (mode == "naive") {
        long long t1 = naive_emb(embedding_table, input, offsets);
        cout << t1 << " microseconds.\n";
    } 
    else if (mode == "prefetch") {
        for (size_t i = 0; i < embedding_table.size(); i += 16) {
            _mm_clflush(&embedding_table[i]);
        }
        _mm_mfence();
        long long t2 = run_with_prefetching(embedding_table, input, offsets);
        cout << t2 << " microseconds.\n";
    }
    else if (mode == "simd") {
        long long t3 = run_with_simd(embedding_table, input, offsets);
        cout << t3 << " microseconds.\n";
    } 
    else if (mode == "prefetch_simd") {
        for (size_t i = 0; i < embedding_table.size(); i += 16) {
            _mm_clflush(&embedding_table[i]);
        }
        _mm_mfence();
        long long t4 = run_with_prefetching_simd(embedding_table, input, offsets);
        cout << t4 << " microseconds.\n";
    } 
    else {
        // default: run all
        long long t1 = naive_emb(embedding_table, input, offsets);
        for (size_t i = 0; i < embedding_table.size(); i += 16) { _mm_clflush(&embedding_table[i]); }
        _mm_mfence();
        long long t2 = run_with_prefetching(embedding_table, input, offsets);
        long long t3 = run_with_simd(embedding_table, input, offsets);
        long long t4 = run_with_prefetching_simd(embedding_table, input, offsets);

        cout << "\n\nSpeedup (prefetch) = " << (double)t1/t2 << "x\n";
        cout << "Speedup (simd) = " << (double)t1/t3 << "x\n";
        cout << "Speedup (prefetch+simd) = " << (double)t1/t4 << "x\n";
    }
    return 0;
}
