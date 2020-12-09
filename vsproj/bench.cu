#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef XOR_MAX_ITERATIONS
#define XOR_MAX_ITERATIONS 100 // probabillity of success should always be > 0.5 so 100 iterations is highly unlikely
#endif 

/**
 * We need a decent random number generator.
 **/

 // returns random number, modifies the seed
static inline uint64_t xor_rng_splitmix64(uint64_t* seed) {
    uint64_t z = (*seed += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/**
 * xor8 is the recommended default, no more than
 * a 0.3% false-positive probability.
 */
typedef struct xor8_s {
    uint64_t seed;
    uint64_t blockLength;
    uint8_t
        * fingerprints; // after xor8_allocate, will point to 3*blockLength values
} xor8_t;

struct xor_xorset_s {
    // split xormask into two 32-bit fields due to atomicXor 32-bit limitation
    uint32_t xormask1;
    uint32_t xormask2;
    uint32_t count;
    uint32_t layer;
};

typedef struct xor_xorset_s xor_xorset_t;

struct xor_keyindex_s {
    uint64_t hash;
    uint32_t index;
};

typedef struct xor_keyindex_s xor_keyindex_t;

struct xor_hashes_s {
    uint64_t h;
    uint32_t h0;
    uint32_t h1;
    uint32_t h2;
};

typedef struct xor_hashes_s xor_hashes_t;

static inline uint64_t xor_murmur64(uint64_t h) {
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    return h;
}

static inline uint64_t xor_mix_split(uint64_t key, uint64_t seed) {
    return xor_murmur64(key + seed);
}

static inline uint64_t xor_rotl64(uint64_t n, unsigned int c) {
    return (n << (c & 63)) | (n >> ((-c) & 63));
}

static inline uint32_t xor_reduce(uint32_t hash, uint32_t n) {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    return (uint32_t)(((uint64_t)hash * n) >> 32);
}

static inline uint64_t xor_fingerprint(uint64_t hash) {
    return hash ^ (hash >> 32);
}

// report memory usage
static inline size_t xor8_size_in_bytes(const xor8_t* filter) {
    return 3 * filter->blockLength * sizeof(uint8_t) + sizeof(xor8_t);
}

// release memory
static inline void xor8_free(xor8_t* filter) {
    cudaFree(filter->fingerprints);
    // free(filter->fingerprints);
    filter->fingerprints = NULL;
    filter->blockLength = 0;
    cudaFree(filter);
}

// allocate enough capacity for a set containing up to 'size' elements
// caller is responsible to call xor8_free(filter)
static inline bool xor8_allocate(uint32_t size, xor8_t* filter) {
    size_t capacity = 32 + 1.23 * size;
    capacity = capacity / 3 * 3;
    // filter->fingerprints = (uint8_t*) malloc(capacity * sizeof(uint8_t));
    cudaMallocManaged(&filter->fingerprints, capacity * sizeof(uint8_t));
    if (filter->fingerprints != NULL) {
        filter->blockLength = capacity / 3;
        memset(filter->fingerprints, 0, capacity * sizeof(uint8_t));
        return true;
    }
    else {
        return false;
    }
}

__device__
static inline uint64_t d_xor_fingerprint(uint64_t hash) {
    return hash ^ (hash >> 32);
}

__device__
static inline uint64_t d_xor_murmur64(uint64_t h) {
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    return h;
}

__device__
static inline uint32_t d_xor_reduce(uint32_t hash, uint32_t n) {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    return (uint32_t)(((uint64_t)hash * n) >> 32);
}

__device__
static inline uint64_t d_xor_rotl64(uint64_t n, unsigned int c) {
    return (n << (c & 63)) | (n >> ((-c) & 63));
}

__device__
static inline uint64_t d_xor_mix_split(uint64_t key, uint64_t seed) {
    return d_xor_murmur64(key + seed);
}

__device__ 
static inline uint32_t d_xor8_get_h0(uint64_t hash, const xor8_t* filter) {
    uint32_t r0 = (uint32_t)hash;
    return d_xor_reduce(r0, filter->blockLength);
}
__device__ 
static inline uint32_t d_xor8_get_h1(uint64_t hash, const xor8_t* filter) {
    uint32_t r1 = (uint32_t)d_xor_rotl64(hash, 21);
    return d_xor_reduce(r1, filter->blockLength);
}
__device__ 
static inline uint32_t d_xor8_get_h2(uint64_t hash, const xor8_t* filter) {
    uint32_t r2 = (uint32_t)d_xor_rotl64(hash, 42);
    return d_xor_reduce(r2, filter->blockLength);
}

__device__
static inline xor_hashes_t d_xor8_get_h0_h1_h2(uint64_t k, const xor8_t* filter) {
    uint64_t hash = d_xor_mix_split(k, filter->seed);
    xor_hashes_t answer;
    answer.h = hash;
    uint32_t r0 = (uint32_t)hash;
    uint32_t r1 = (uint32_t)d_xor_rotl64(hash, 21);
    uint32_t r2 = (uint32_t)d_xor_rotl64(hash, 42);

    answer.h0 = d_xor_reduce(r0, filter->blockLength);
    answer.h1 = d_xor_reduce(r1, filter->blockLength);
    answer.h2 = d_xor_reduce(r2, filter->blockLength);
    return answer;
}

__global__
void peelSet0(uint32_t* d_index, xor_xorset_t* sets0, xor_xorset_t* sets1, xor_xorset_t* sets2, 
    xor8_t* filter, size_t* layer, bool* pureCell) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < filter->blockLength ; i += stride) {
        if (sets0[i].count == 1) {
            sets0[i].count--;
            uint64_t hash = ((uint64_t)sets0[i].xormask1) << 32 | sets0[i].xormask2;
            sets0[i].layer = *layer; // determine peel ordering
            *pureCell = true; // race condition but should be safe

            uint32_t h1 = d_xor8_get_h1(hash, filter);
            uint32_t h2 = d_xor8_get_h2(hash, filter);

            atomicXor(&sets1[h1].xormask1, sets0[i].xormask1);
            atomicXor(&sets1[h1].xormask2, sets0[i].xormask2);
            atomicSub(&sets1[h1].count, 1);

            atomicXor(&sets2[h2].xormask1, sets0[i].xormask1);
            atomicXor(&sets2[h2].xormask2, sets0[i].xormask2);
            atomicSub(&sets2[h2].count, 1);
        }
    }
}

__global__
void peelSet1(uint32_t* d_index, xor_xorset_t* sets0, xor_xorset_t* sets1, xor_xorset_t* sets2,
    xor8_t* filter, size_t* layer, bool* pureCell) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < filter->blockLength; i += stride) {
        if (sets1[i].count == 1) {
            sets1[i].count--;
            uint64_t hash = ((uint64_t)sets1[i].xormask1) << 32 | sets1[i].xormask2;
            sets1[i].layer = *layer; // determine peel ordering
            *pureCell = true; // race condition but should be safe

            uint32_t h0 = d_xor8_get_h0(hash, filter);
            uint32_t h2 = d_xor8_get_h2(hash, filter);

            atomicXor(&sets0[h0].xormask1, sets1[i].xormask1);
            atomicXor(&sets0[h0].xormask2, sets1[i].xormask2);
            atomicSub(&sets0[h0].count, 1);

            atomicXor(&sets2[h2].xormask1, sets1[i].xormask1);
            atomicXor(&sets2[h2].xormask2, sets1[i].xormask2);
            atomicSub(&sets2[h2].count, 1);
        }
    }
}

__global__
void peelSet2(uint32_t* d_index, xor_xorset_t* sets0, xor_xorset_t* sets1, xor_xorset_t* sets2,
    xor8_t* filter, size_t* layer, bool* pureCell) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < filter->blockLength; i += stride) {
        if (sets2[i].count == 1) {
            sets2[i].count--;
            uint64_t hash = ((uint64_t)sets2[i].xormask1) << 32 | sets2[i].xormask2;
            sets2[i].layer = *layer; // determine peel ordering
            *pureCell = true; // race condition but should be safe

            uint32_t h0 = d_xor8_get_h0(hash, filter);
            uint32_t h1 = d_xor8_get_h1(hash, filter);

            atomicXor(&sets0[h0].xormask1, sets2[i].xormask1);
            atomicXor(&sets0[h0].xormask2, sets2[i].xormask2);
            atomicSub(&sets0[h0].count, 1);

            atomicXor(&sets1[h1].xormask1, sets2[i].xormask1);
            atomicXor(&sets1[h1].xormask2, sets2[i].xormask2);
            atomicSub(&sets1[h1].count, 1);
        }
    }
}

__global__
void insertKeys(const uint64_t* keys, uint32_t size, xor_xorset_t* sets0, 
    xor_xorset_t* sets1, xor_xorset_t* sets2, xor8_t* filter) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        uint64_t key = keys[i];
        xor_hashes_t hs = d_xor8_get_h0_h1_h2(key, filter);

        uint32_t hsh1 = (uint32_t)(hs.h >> 32);
        uint32_t hsh2 = (uint32_t)hs.h;

        atomicXor(&sets0[hs.h0].xormask1, hsh1);
        atomicXor(&sets0[hs.h0].xormask2, hsh2);
        atomicAdd(&sets0[hs.h0].count, 1);

        atomicXor(&sets1[hs.h1].xormask1, hsh1);
        atomicXor(&sets1[hs.h1].xormask2, hsh2);
        atomicAdd(&sets1[hs.h1].count, 1);

        atomicXor(&sets2[hs.h2].xormask1, hsh1);
        atomicXor(&sets2[hs.h2].xormask2, hsh2);
        atomicAdd(&sets2[hs.h2].count, 1);
    }

}

__global__
void assign(xor8_t* filter, xor_xorset_t* sets, size_t layer, size_t arrayLength, 
    uint8_t* fingerprints0, uint8_t* fingerprints1, uint8_t* fingerprints2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < arrayLength; i += stride) {
        if (sets[i].layer == layer) {
            uint64_t key_hash = ((uint64_t)sets[i].xormask1) << 32 | sets[i].xormask2;
            uint64_t val = d_xor_fingerprint(key_hash);
            if (i < filter->blockLength) {
                val ^= fingerprints1[d_xor8_get_h1(key_hash, filter)] ^ fingerprints2[d_xor8_get_h2(key_hash, filter)];
            }
            else if (i < 2 * filter->blockLength) {
                val ^= fingerprints0[d_xor8_get_h0(key_hash, filter)] ^ fingerprints2[d_xor8_get_h2(key_hash, filter)];
            }
            else {
                val ^= fingerprints0[d_xor8_get_h0(key_hash, filter)] ^ fingerprints1[d_xor8_get_h1(key_hash, filter)];
            }
            filter->fingerprints[i] = val;
        }
    }
}

//
// construct the filter, returns true on success, false on failure.
// most likely, a failure is due to too high a memory usage
// size is the number of keys
// The caller is responsable for calling xor8_allocate(size,filter) before.
// The caller is responsible to ensure that there are no duplicated keys.
// The inner loop will run up to XOR_MAX_ITERATIONS times (default on 100),
// it should never fail, except if there are duplicated keys. If it fails,
// a return value of false is provided.
//
bool xor8_populate(const uint64_t* keys, uint32_t size, xor8_t* filter) {
    uint64_t rng_counter = 1;
    filter->seed = xor_rng_splitmix64(&rng_counter);
    size_t arrayLength = filter->blockLength * 3; // size of the backing array
    size_t blockLength = filter->blockLength;

    xor_xorset_t* sets; 
    cudaError_t errSets = cudaMallocManaged(&sets, arrayLength * sizeof(xor_xorset_t));

    if (errSets != cudaSuccess) {
        fprintf(stderr, "Failed to allocate unified memory (error code %s)!\n", 
            cudaGetErrorString(errSets));
        return false;
    }
    xor_xorset_t* sets0 = sets;
    xor_xorset_t* sets1 = sets + blockLength;
    xor_xorset_t* sets2 = sets + 2 * blockLength;

    int iterations = 0;

    size_t* layer;
    cudaMallocManaged(&layer, sizeof(size_t));

    // parallelism config
    int blockSize = 128;
    int numBlocks = (filter->blockLength + blockSize - 1) / blockSize;

    while (true) {
        iterations++;
        if (iterations > XOR_MAX_ITERATIONS) {
            fprintf(stderr, "Too many iterations. Are all your keys unique?");
            cudaFree(sets);
            return false;
        }

        // inserting keys into temporary array H
        memset(sets, 0, sizeof(xor_xorset_t) * arrayLength);

        insertKeys << <numBlocks, blockSize >> > (keys, size, sets0, sets1, sets2, filter);
        cudaDeviceSynchronize();

        // Start parallelization here
        
        bool* pureCell;
        cudaMallocManaged(&pureCell, sizeof(bool));
        *pureCell = false;
        *layer = 1;
        size_t old_layer = 0;
        
        while (old_layer != *layer) {
            old_layer = *layer;

            peelSet0 << <numBlocks, blockSize>> > (NULL, sets0, sets1, sets2, filter, layer, pureCell);
            cudaDeviceSynchronize();

            if (*pureCell == true) {
                *layer = *layer + 1;
                *pureCell = false;
            }

            peelSet1 << <numBlocks, blockSize>> > (NULL, sets0, sets1, sets2, filter, layer, pureCell);
            cudaDeviceSynchronize();

            if (*pureCell == true) {
                *layer = *layer + 1;
                *pureCell = false;
            }

            peelSet2 << <numBlocks, blockSize>> > (NULL, sets0, sets1, sets2, filter, layer, pureCell);
            cudaDeviceSynchronize();

            if (*pureCell == true) {
                *layer = *layer + 1;
                *pureCell = false;
            }
        }

        cudaFree(pureCell);

        size_t recover_cnt = 0;
        for (int i = 0; i < arrayLength; i++) {
            if (sets[i].layer > 0) {
                recover_cnt++;
            }
        }

        if (recover_cnt == size) {
            //success
            break;
        }

        filter->seed = xor_rng_splitmix64(&rng_counter);
    }
    uint8_t* fingerprints0 = filter->fingerprints;
    uint8_t* fingerprints1 = filter->fingerprints + blockLength;
    uint8_t* fingerprints2 = filter->fingerprints + 2 * blockLength;

    size_t layer_size = *layer;
    while (layer_size > 0) {
        assign << <numBlocks, blockSize >> > (filter, sets, layer_size, arrayLength, fingerprints0, fingerprints1, fingerprints2);
        layer_size--;
    }
    cudaDeviceSynchronize();
    
    cudaFree(layer);
    cudaFree(sets);
    return true;
}

// Report if the key is in the set, with false positive rate.
static inline bool xor8_contain(uint64_t key, const xor8_t* filter) {
    uint64_t hash = xor_mix_split(key, filter->seed);
    uint8_t f = xor_fingerprint(hash);
    uint32_t r0 = (uint32_t)hash;
    uint32_t r1 = (uint32_t)xor_rotl64(hash, 21);
    uint32_t r2 = (uint32_t)xor_rotl64(hash, 42);
    uint32_t h0 = xor_reduce(r0, filter->blockLength);
    uint32_t h1 = xor_reduce(r1, filter->blockLength) + filter->blockLength;
    uint32_t h2 = xor_reduce(r2, filter->blockLength) + 2 * filter->blockLength;
    return f == (filter->fingerprints[h0] ^ filter->fingerprints[h1] ^
        filter->fingerprints[h2]);
}

bool testxor8(size_t size) {
    printf("testing xor8 ");
    printf("size = %zu \n", size);

    xor8_t* filter;
    cudaMallocManaged(&filter, sizeof(xor8_t));

    xor8_allocate(size, filter);
    // we need some set of values
    uint64_t* big_set;
    cudaMallocManaged(&big_set, sizeof(uint64_t) * size);

    for (size_t i = 0; i < size; i++) {
        big_set[i] = i; // we use contiguous values
    }
    // we construct the filter
    xor8_populate(big_set, size, filter); // warm the cache
    for (size_t times = 0; times < 5; times++) {
        clock_t t;
        t = clock();
        xor8_populate(big_set, size, filter);
        t = clock() - t;
        double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
        printf("It took %f seconds to build an index over %zu values. \n",
            time_taken, size);
    }
    xor8_free(filter);
    cudaFree(big_set);
    return true;
}

bool testxor8() {
    printf("testing xor8\n");

    xor8_t* filter;
    cudaMallocManaged(&filter, sizeof(xor8_t));
    size_t size = 10000000;
    xor8_allocate(size, filter);
    // we need some set of values
    uint64_t* big_set;
    cudaMallocManaged(&big_set, sizeof(uint64_t) * size);

    for (size_t i = 0; i < size; i++) {
        big_set[i] = i; // we use contiguous values
    }
    // we construct the filter
    xor8_populate(big_set, size, filter);
    for (size_t i = 0; i < size; i++) {
        if (!xor8_contain(big_set[i], filter)) {
            printf("bug!\n");
            return false;
        }
    }

    size_t random_matches = 0;
    size_t trials = 10000000; //(uint64_t)rand() << 32 + rand()
    for (size_t i = 0; i < trials; i++) {
        uint64_t random_key = ((uint64_t)rand() << 32) + rand();
        if (xor8_contain(random_key, filter)) {
            if (random_key >= size) {
                random_matches++;
            }
        }
    }
    printf("fpp %3.10f (estimated) \n", random_matches * 1.0 / trials);
    printf("bits per entry %3.1f\n", xor8_size_in_bytes(filter) * 8.0 / size);
    xor8_free(filter);
    cudaFree(big_set);
    return true;
}

/* int main() {
    for (size_t s = 10000000; s <= 10000000; s *= 10) {
        // testxor8(s);
        testxor8();

        printf("\n");
    }
} */