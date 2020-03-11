/* Cuda memcheck snippets from HW3 
 * http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
 * Some other helper functions was used from CUB library
 */


#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "mersenne.h"
#include <bits/stdc++.h> 
#include <sys/time.h> 

// Macro 
#define MAX(a, b) (((b) > (a)) ? (b) : (a))

#define AssertEquals(a, b) if ((a) != (b)) { std::cerr << "\n(" << __FILE__ << ": " << __LINE__ << ")\n"; exit(1);}

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                              \
    cudaError err = call;                                                 \
    if( cudaSuccess != err) {                                             \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
                  __FILE__, __LINE__, cudaGetErrorString( err) );         \
      exit(EXIT_FAILURE);                                                 \
      } } while (0)
  
#define CUDA_SAFE_CALL( call) do {                                      \
    CUDA_SAFE_CALL_NO_SYNC(call);                                         \
    cudaError err = cudaDeviceSynchronize();                              \
    if( cudaSuccess != err) {                                             \
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                   __FILE__, __LINE__, cudaGetErrorString( err) );        \
       exit(EXIT_FAILURE);                                                \
       } } while (0)



using namespace std;

// Functions

// Argument handler
void usage(char** argv) {
    fprintf(stderr, "Usage: %s "
            "<input items> "
            "<verbose> "
            "<device-id> "
            "\n", argv[0]);
    exit(1);
}
void argsHandler (int argc, char** argv, ulong* num_items, bool* g_verbose, int* deviceid){
    if (argc < 2 || argc > 4) {
        usage(argv);
    }
    *num_items = atoi(argv[1]);
    if (argc >= 3)
        *g_verbose = atoi(argv[2]);
    if (argc >= 4)
    *deviceid = atoi(argv[3]);
    return;
}

void argsHandlerThresh (int argc, char** argv, ulong* num_items, bool* g_verbose, int* deviceid, int* thresh){
    if (argc < 2 || argc > 5) {
        usage(argv);
    }
    *num_items = atoi(argv[1]);
    if (argc >= 3)
        *thresh = atoi(argv[2]);   
    if (argc >= 4)
        *g_verbose = atoi(argv[3]);
    if (argc >= 5)
    *deviceid = atoi(argv[4]);
    return;
}

/**
 * Simple Fucntion to check if value is NaN
 */
template <typename T>
bool IsNaN(T val) { return val!=val; }

/**
 * Simple Fucntion to check if number is power of 2
 */
bool IsPowerOfTwo(ulong x) { return (x != 0) && ((x & (x - 1)) == 0); }

/**
 * Helper for casting character types to integers for cout printing
 */
 template <typename T>
 T CoutCast(T val) { return val; }

struct Pair
{
    float   key;
    bool operator<(const Pair &b) const
    {
        if (key < b.key)
            return true;
        if (key > b.key)
            return false;
        // Return true if key is negative zero and b.key is positive zero
        unsigned int key_bits   = *reinterpret_cast<unsigned*>(const_cast<float*>(&key));
        unsigned int b_key_bits = *reinterpret_cast<unsigned*>(const_cast<float*>(&b.key));
        unsigned int HIGH_BIT   = 1u << 31;
        return ((key_bits & HIGH_BIT) != 0) && ((b_key_bits & HIGH_BIT) == 0);
    }
};

/**
 * A Helper function to generate random bits from Mersenne prime algorithm
 */
int g_num_rand_samples = 0;
template <typename K>
void RandomBits(
    K &key,
    int entropy_reduction = 0,
    int begin_bit = 0,
    int end_bit = sizeof(K) * 8)
{
    const int NUM_BYTES = sizeof(K);
    const int WORD_BYTES = sizeof(unsigned int);
    const int NUM_WORDS = (NUM_BYTES + WORD_BYTES - 1) / WORD_BYTES;

    unsigned int word_buff[NUM_WORDS];

    if (entropy_reduction == -1)
    {
        memset((void *) &key, 0, sizeof(key));
        return;
    }

    if (end_bit < 0)
        end_bit = sizeof(K) * 8;

    while (true) 
    {
        // Generate random word_buff
        for (int j = 0; j < NUM_WORDS; j++)
        {
            int current_bit = j * WORD_BYTES * 8;

            unsigned int word = 0xffffffff;
            word &= 0xffffffff << MAX(0, begin_bit - current_bit);
            word &= 0xffffffff >> MAX(0, (current_bit + (WORD_BYTES * 8)) - end_bit);

            for (int i = 0; i <= entropy_reduction; i++)
            {
                // Grab some of the higher bits from rand (better entropy, supposedly)
                word &= mersenne::genrand_int32();
                g_num_rand_samples++;                
            }

            word_buff[j] = word;
        }

        memcpy(&key, word_buff, sizeof(K));

        K copy = key;
        if (!IsNaN(copy))
            break;          // avoids NaNs when generating random floating point numbers
    }
}

/**
 * A function to Display the data on console
 */
template <typename InputIteratorT>
void DisplayResults(
    InputIteratorT h_data,
    size_t num_items)
{
    // Display data
    for (ulong i = 0; i < ulong(num_items); i++)
    {
        std::cout << CoutCast(h_data[i]);
        if (i < ulong(num_items)-1)
            std::cout << ", ";
    }
    printf("\n");
}

/**
 * A function to Display the data on console only head and tail
 */
template <typename InputIteratorT>
void DisplayResultsHT(
    InputIteratorT h_data,
    size_t num_items,
    int thresh)
{
    int flag = 0;
    // Display data only head and tail
    for (int i = 0; i < 2 * thresh - 1; i++){
        if (i == num_items/2 || (int)num_items - (2 * thresh - i - 1) == num_items/2)
            if (flag == 0)
                flag = 1;
            else
                continue;
            printf("for i=%d: data=%f\n", (i < thresh) ? i : (int)num_items - ( 2 * thresh - i - 1) , (i < thresh) ? h_data[i] : h_data[(int)num_items - (2 * thresh - i - 1)]);
    }
    printf("\n");
}

/**
 * A Helper function to Initalize random elements on host
 */
void Initialize(
    float           *h_keys,
    float           *h_reference_keys,
    ulong             num_items,
    bool            g_verbose)
{
    struct timespec start, end; 
    Pair *h_pairs = new Pair[num_items];
    for (int i = 0; i < num_items; ++i)
    {
        RandomBits(h_keys[i]);
        h_pairs[i].key    = h_keys[i];
    }
    if (g_verbose)
    {
        printf("Input keys:\n");
        DisplayResults(h_keys, num_items);
        printf("\n\n");
    }
    clock_gettime(CLOCK_MONOTONIC, &start); 
    ios_base::sync_with_stdio(false);
    std::sort(h_pairs, h_pairs + num_items);
    clock_gettime(CLOCK_MONOTONIC, &end); 
    // Calculating total time taken by the program. 
    double time_taken; 
    time_taken = (end.tv_sec - start.tv_sec) * 1e9; 
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9; 
  
    std::cout << "Time taken by std::sort on CPU is : " << fixed 
         << time_taken * 1.0e3 << setprecision(9); 
    std::cout << " msec" << " \t and" ; 
        std::cout << "Speed by program on CPU is : " << fixed 
         << 1.0e-6 * (double)num_items/time_taken << setprecision(5); 
    std::cout << " MElements/s" << endl; 
    for (int i = 0; i < num_items; ++i)
    {
        h_reference_keys[i]     = h_pairs[i].key;
    }
    delete[] h_pairs;
}

/**
 * A Helper function to Compare Results on host (computed must be already copied to host)
 */
template <typename S, typename T, typename OffsetT>
int CompareResults(T* computed, S* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            std::cout << "INCORRECT: [" << i << "]: "
                << CoutCast(computed[i]) << " != "
                << CoutCast(reference[i]);
            return 1;
        }
    }
    return 0;
}
