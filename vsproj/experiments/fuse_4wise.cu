#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define FUSE_ARITY 3
#define FUSE_SEGMENT_COUNT 100
#define FUSE_SLOTS (FUSE_SEGMENT_COUNT + FUSE_ARITY - 1)

#ifndef XOR_MAX_ITERATIONS
#define XOR_MAX_ITERATIONS 100 // probabillity of success should always be > 0.5 so 100 iterations is highly unlikely
#endif 
/**
 * fuse8 is the recommended default, no more than
 * a 0.3% false-positive probability.
 */
typedef struct fuse8_s {
    uint64_t seed;
    uint64_t segmentLength; // = slotCount  / FUSE_SLOTS
    uint8_t
        * fingerprints; // after fuse8_allocate, will point to 3*blockLength values
} fuse8_t;

struct fuse_fuseset_s {
    // Note: 64-bit width not supported on Amazon EC2 GPU instance
    // uint64_t fusemask;
    uint32_t fusemask1;
    uint32_t fusemask2;
    uint32_t count;
    uint32_t layer;
};

typedef struct fuse_fuseset_s fuse_fuseset_t;

struct fuse_keyindex_s {
    uint64_t hash;
    uint32_t index;
};

typedef struct fuse_keyindex_s fuse_keyindex_t;

struct fuse_hashes_s {
    uint64_t h;
    uint32_t h0;
    uint32_t h1;
    uint32_t h2;
    uint32_t h3;
};

typedef struct fuse_hashes_s fuse_hashes_t;

struct fuse_h0h1h2h3_s {
    uint32_t h0;
    uint32_t h1;
    uint32_t h2;
    uint32_t h3;
};

typedef struct fuse_h0h1h2h3_s fuse_h0h1h2_t;

static inline uint64_t fuse_murmur64(uint64_t h) {
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    return h;
}

static inline uint64_t fuse_rotl64(uint64_t n, unsigned int c) {
    return (n << (c & 63)) | (n >> ((-c) & 63));
}

static inline uint32_t fuse_reduce(uint32_t hash, uint32_t n) {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    return (uint32_t)(((uint64_t)hash * n) >> 32);
}

// returns random number, modifies the seed
static inline uint64_t fuse_rng_splitmix64(uint64_t* seed) {
    uint64_t z = (*seed += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static inline uint64_t fuse_fingerprint(uint64_t hash) {
    return hash ^ (hash >> 32);
}

// report memory usage
static inline size_t fuse8_size_in_bytes(const fuse8_t* filter) {
    return FUSE_SLOTS * filter->segmentLength * sizeof(uint8_t) + sizeof(fuse8_t);
}

static inline uint64_t fuse_mix_split(uint64_t key, uint64_t seed) {
    return fuse_murmur64(key + seed);
}

__device__
static inline uint64_t d_fuse_murmur64(uint64_t h) {
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    return h;
}

__device__
static inline uint64_t d_fuse_fingerprint(uint64_t hash) {
    return hash ^ (hash >> 32);
}

__device__
static inline uint64_t d_fuse_rotl64(uint64_t n, unsigned int c) {
    return (n << (c & 63)) | (n >> ((-c) & 63));
}

__device__
static inline uint32_t d_fuse_reduce(uint32_t hash, uint32_t n) {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    return (uint32_t)(((uint64_t)hash * n) >> 32);
}

__device__
static inline uint64_t d_fuse_mix_split(uint64_t key, uint64_t seed) {
    return d_fuse_murmur64(key + seed);
}

__device__
static inline fuse_hashes_t d_fuse8_get_h0_h1_h2_h3(uint64_t k, const fuse8_t* filter) {
    uint64_t hash = d_fuse_mix_split(k, filter->seed);
    fuse_hashes_t answer;
    answer.h = hash;
    uint32_t r0 = (uint32_t)hash;
    uint32_t r1 = (uint32_t)d_fuse_rotl64(hash, 21);
    uint32_t r2 = (uint32_t)d_fuse_rotl64(hash, 42);
    uint32_t r3 = (0xBF58476D1CE4E5B9 * hash) >> 32;
    uint32_t seg = d_fuse_reduce(r0, FUSE_SEGMENT_COUNT);
    answer.h0 = (seg + 0) * filter->segmentLength + d_fuse_reduce(r1, filter->segmentLength);
    answer.h1 = (seg + 1) * filter->segmentLength + d_fuse_reduce(r2, filter->segmentLength);
    answer.h2 = (seg + 2) * filter->segmentLength + d_fuse_reduce(r3, filter->segmentLength);
    return answer;
}

__device__
static inline fuse_h0h1h2_t d_fuse8_get_just_h0_h1_h2(uint64_t hash,
    const fuse8_t* filter) {
    fuse_h0h1h2_t answer;
    uint32_t r0 = (uint32_t)hash;
    uint32_t r1 = (uint32_t)d_fuse_rotl64(hash, 21);
    uint32_t r2 = (uint32_t)d_fuse_rotl64(hash, 42);
    uint32_t r3 = (0xBF58476D1CE4E5B9 * hash) >> 32;
    uint32_t seg = d_fuse_reduce(r0, FUSE_SEGMENT_COUNT);
    answer.h0 = (seg + 0) * filter->segmentLength + d_fuse_reduce(r1, filter->segmentLength);
    answer.h1 = (seg + 1) * filter->segmentLength + d_fuse_reduce(r2, filter->segmentLength);
    answer.h2 = (seg + 2) * filter->segmentLength + d_fuse_reduce(r3, filter->segmentLength);
    return answer;
}

__global__
void insert_keys(const uint64_t* keys, uint32_t size, fuse_fuseset_t* sets, fuse8_t* filter) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        uint64_t key = keys[i];
        fuse_hashes_t hs = d_fuse8_get_h0_h1_h2(key, filter);
        uint32_t hsh1 = (uint32_t)(hs.h >> 32);
        uint32_t hsh2 = (uint32_t)hs.h;
        atomicXor(&sets[hs.h0].fusemask1, hsh1);
        atomicXor(&sets[hs.h0].fusemask2, hsh2);
        atomicAdd(&sets[hs.h0].count, 1);
        atomicXor(&sets[hs.h1].fusemask1, hsh1);
        atomicXor(&sets[hs.h1].fusemask2, hsh2);
        atomicAdd(&sets[hs.h1].count, 1);
        atomicXor(&sets[hs.h2].fusemask1, hsh1);
        atomicXor(&sets[hs.h2].fusemask2, hsh2);
        atomicAdd(&sets[hs.h2].count, 1);
    }
}

__global__
void peel_set0(fuse_fuseset_t* sets, fuse_fuseset_t* sets0, fuse_fuseset_t* sets1, 
    fuse_fuseset_t* sets2, fuse8_t* filter, size_t* layer, bool* pureCell) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < filter->segmentLength * (FUSE_SLOTS / 3); i += stride) {
        if (sets0[i].count == 1) {
            uint32_t sets_idx = ((i / filter->segmentLength) * 3) * filter->segmentLength + i % filter->segmentLength;
            sets0[i].count--;
            sets[sets_idx].count--;
            uint64_t hash = ((uint64_t)sets0[i].fusemask1) << 32 | sets0[i].fusemask2;
            sets0[i].layer = *layer;
            sets[sets_idx].layer = *layer;
            *pureCell = true; // race condition but should be safe
            fuse_h0h1h2_t hs = d_fuse8_get_just_h0_h1_h2(hash, filter);

            if (hs.h0 != sets_idx) {
                atomicXor(&sets[hs.h0].fusemask1, sets0[i].fusemask1);
                atomicXor(&sets[hs.h0].fusemask2, sets0[i].fusemask2);
                atomicSub(&sets[hs.h0].count, 1);

                uint32_t h0_seg = hs.h0 / filter->segmentLength;
                uint32_t h0_idx = (h0_seg / 3) * filter->segmentLength + hs.h0 % filter->segmentLength;
                fuse_fuseset_t* h0_set;
                if (h0_seg % FUSE_ARITY == 0) {
                    h0_set = sets0;
                }
                else if (h0_seg % FUSE_ARITY == 1) {
                    h0_set = sets1;
                }
                else {
                    h0_set = sets2;
                }
                atomicXor(&h0_set[h0_idx].fusemask1, sets0[i].fusemask1);
                atomicXor(&h0_set[h0_idx].fusemask2, sets0[i].fusemask2);
                atomicSub(&h0_set[h0_idx].count, 1);
            }

            if (hs.h1 != sets_idx) {
                atomicXor(&sets[hs.h1].fusemask1, sets0[i].fusemask1);
                atomicXor(&sets[hs.h1].fusemask2, sets0[i].fusemask2);
                atomicSub(&sets[hs.h1].count, 1);

                uint32_t h1_seg = hs.h1 / filter->segmentLength;
                uint32_t h1_idx = (h1_seg / 3) * filter->segmentLength + hs.h1 % filter->segmentLength;
                fuse_fuseset_t* h1_set;
                if (h1_seg % FUSE_ARITY == 0) {
                    h1_set = sets0;
                }
                else if (h1_seg % FUSE_ARITY == 1) {
                    h1_set = sets1;
                }
                else {
                    h1_set = sets2;
                }
                atomicXor(&h1_set[h1_idx].fusemask1, sets0[i].fusemask1);
                atomicXor(&h1_set[h1_idx].fusemask2, sets0[i].fusemask2);
                atomicSub(&h1_set[h1_idx].count, 1);
            }

            if (hs.h2 != sets_idx) {
                atomicXor(&sets[hs.h2].fusemask1, sets0[i].fusemask1);
                atomicXor(&sets[hs.h2].fusemask2, sets0[i].fusemask2);
                atomicSub(&sets[hs.h2].count, 1);

                uint32_t h2_seg = hs.h2 / filter->segmentLength;
                uint32_t h2_idx = (h2_seg / 3) * filter->segmentLength + hs.h2 % filter->segmentLength;
                fuse_fuseset_t* h2_set;
                if (h2_seg % FUSE_ARITY == 0) {
                    h2_set = sets0;
                }
                else if (h2_seg % FUSE_ARITY == 1) {
                    h2_set = sets1;
                }
                else {
                    h2_set = sets2;
                }
                atomicXor(&h2_set[h2_idx].fusemask1, sets0[i].fusemask1);
                atomicXor(&h2_set[h2_idx].fusemask2, sets0[i].fusemask2);
                atomicSub(&h2_set[h2_idx].count, 1);
            }

        }
    }
}

__global__
void peel_set1(fuse_fuseset_t* sets, fuse_fuseset_t* sets0, fuse_fuseset_t* sets1,
    fuse_fuseset_t* sets2, fuse8_t* filter, size_t* layer, bool* pureCell) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < filter->segmentLength * (FUSE_SLOTS / 3); i += stride) {
        if (sets1[i].count == 1) {
            uint32_t sets_idx = ((i / filter->segmentLength) * 3 + 1) * filter->segmentLength + i % filter->segmentLength;
            sets1[i].count--;
            sets[sets_idx].count--;
            uint64_t hash = ((uint64_t)sets1[i].fusemask1) << 32 | sets1[i].fusemask2;
            sets1[i].layer = *layer;
            sets[sets_idx].layer = *layer;
            *pureCell = true; // race condition but should be safe
            fuse_h0h1h2_t hs = d_fuse8_get_just_h0_h1_h2(hash, filter);

            if (hs.h0 != sets_idx) {
                atomicXor(&sets[hs.h0].fusemask1, sets1[i].fusemask1);
                atomicXor(&sets[hs.h0].fusemask2, sets1[i].fusemask2);
                atomicSub(&sets[hs.h0].count, 1);

                uint32_t h0_seg = hs.h0 / filter->segmentLength;
                uint32_t h0_idx = (h0_seg / 3) * filter->segmentLength + hs.h0 % filter->segmentLength;
                fuse_fuseset_t* h0_set;
                if (h0_seg % FUSE_ARITY == 0) {
                    h0_set = sets0;
                }
                else if (h0_seg % FUSE_ARITY == 1) {
                    h0_set = sets1;
                }
                else {
                    h0_set = sets2;
                }
                atomicXor(&h0_set[h0_idx].fusemask1, sets1[i].fusemask1);
                atomicXor(&h0_set[h0_idx].fusemask2, sets1[i].fusemask2);
                atomicSub(&h0_set[h0_idx].count, 1);
            }

            if (hs.h1 != sets_idx) {
                atomicXor(&sets[hs.h1].fusemask1, sets1[i].fusemask1);
                atomicXor(&sets[hs.h1].fusemask2, sets1[i].fusemask2);
                atomicSub(&sets[hs.h1].count, 1);

                uint32_t h1_seg = hs.h1 / filter->segmentLength;
                uint32_t h1_idx = (h1_seg / 3) * filter->segmentLength + hs.h1 % filter->segmentLength;
                fuse_fuseset_t* h1_set;
                if (h1_seg % FUSE_ARITY == 0) {
                    h1_set = sets0;
                }
                else if (h1_seg % FUSE_ARITY == 1) {
                    h1_set = sets1;
                }
                else {
                    h1_set = sets2;
                }
                atomicXor(&h1_set[h1_idx].fusemask1, sets1[i].fusemask1);
                atomicXor(&h1_set[h1_idx].fusemask2, sets1[i].fusemask2);
                atomicSub(&h1_set[h1_idx].count, 1);
            }

            if (hs.h2 != sets_idx) {
                atomicXor(&sets[hs.h2].fusemask1, sets1[i].fusemask1);
                atomicXor(&sets[hs.h2].fusemask2, sets1[i].fusemask2);
                atomicSub(&sets[hs.h2].count, 1);

                uint32_t h2_seg = hs.h2 / filter->segmentLength;
                uint32_t h2_idx = (h2_seg / 3) * filter->segmentLength + hs.h2 % filter->segmentLength;
                fuse_fuseset_t* h2_set;
                if (h2_seg % FUSE_ARITY == 0) {
                    h2_set = sets0;
                }
                else if (h2_seg % FUSE_ARITY == 1) {
                    h2_set = sets1;
                }
                else {
                    h2_set = sets2;
                }
                atomicXor(&h2_set[h2_idx].fusemask1, sets1[i].fusemask1);
                atomicXor(&h2_set[h2_idx].fusemask2, sets1[i].fusemask2);
                atomicSub(&h2_set[h2_idx].count, 1);
            }

        }
    }
}

__global__
void peel_set2(fuse_fuseset_t* sets, fuse_fuseset_t* sets0, fuse_fuseset_t* sets1,
    fuse_fuseset_t* sets2, fuse8_t* filter, size_t* layer, bool* pureCell) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < filter->segmentLength * (FUSE_SLOTS / 3); i += stride) {
        if (sets2[i].count == 1) {
            uint32_t sets_idx = ((i / filter->segmentLength) * 3 + 2) * filter->segmentLength + i % filter->segmentLength;
            sets2[i].count--;
            sets[sets_idx].count--;
            uint64_t hash = ((uint64_t)sets2[i].fusemask1) << 32 | sets2[i].fusemask2;
            sets2[i].layer = *layer;
            sets[sets_idx].layer = *layer;
            *pureCell = true; // race condition but should be safe
            fuse_h0h1h2_t hs = d_fuse8_get_just_h0_h1_h2(hash, filter);

            if (hs.h0 != sets_idx) {
                atomicXor(&sets[hs.h0].fusemask1, sets2[i].fusemask1);
                atomicXor(&sets[hs.h0].fusemask2, sets2[i].fusemask2);
                atomicSub(&sets[hs.h0].count, 1);

                uint32_t h0_seg = hs.h0 / filter->segmentLength;
                uint32_t h0_idx = (h0_seg / 3) * filter->segmentLength + hs.h0 % filter->segmentLength;
                fuse_fuseset_t* h0_set;
                if (h0_seg % FUSE_ARITY == 0) {
                    h0_set = sets0;
                }
                else if (h0_seg % FUSE_ARITY == 1) {
                    h0_set = sets1;
                }
                else {
                    h0_set = sets2;
                }
                atomicXor(&h0_set[h0_idx].fusemask1, sets2[i].fusemask1);
                atomicXor(&h0_set[h0_idx].fusemask2, sets2[i].fusemask2);
                atomicSub(&h0_set[h0_idx].count, 1);
            }

            if (hs.h1 != sets_idx) {
                atomicXor(&sets[hs.h1].fusemask1, sets2[i].fusemask1);
                atomicXor(&sets[hs.h1].fusemask2, sets2[i].fusemask2);
                atomicSub(&sets[hs.h1].count, 1);

                uint32_t h1_seg = hs.h1 / filter->segmentLength;
                uint32_t h1_idx = (h1_seg / 3) * filter->segmentLength + hs.h1 % filter->segmentLength;
                fuse_fuseset_t* h1_set;
                if (h1_seg % FUSE_ARITY == 0) {
                    h1_set = sets0;
                }
                else if (h1_seg % FUSE_ARITY == 1) {
                    h1_set = sets1;
                }
                else {
                    h1_set = sets2;
                }
                atomicXor(&h1_set[h1_idx].fusemask1, sets2[i].fusemask1);
                atomicXor(&h1_set[h1_idx].fusemask2, sets2[i].fusemask2);
                atomicSub(&h1_set[h1_idx].count, 1);
            }

            if (hs.h2 != sets_idx) {
                atomicXor(&sets[hs.h2].fusemask1, sets2[i].fusemask1);
                atomicXor(&sets[hs.h2].fusemask2, sets2[i].fusemask2);
                atomicSub(&sets[hs.h2].count, 1);

                uint32_t h2_seg = hs.h2 / filter->segmentLength;
                uint32_t h2_idx = (h2_seg / 3) * filter->segmentLength + hs.h2 % filter->segmentLength;
                fuse_fuseset_t* h2_set;
                if (h2_seg % FUSE_ARITY == 0) {
                    h2_set = sets0;
                }
                else if (h2_seg % FUSE_ARITY == 1) {
                    h2_set = sets1;
                }
                else {
                    h2_set = sets2;
                }
                atomicXor(&h2_set[h2_idx].fusemask1, sets2[i].fusemask1);
                atomicXor(&h2_set[h2_idx].fusemask2, sets2[i].fusemask2);
                atomicSub(&h2_set[h2_idx].count, 1);
            }

        }
    }
}

__global__
void assign(fuse8_t* filter, fuse_fuseset_t* sets, size_t layer, size_t arrayLength) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < arrayLength; i += stride) {
        if (sets[i].layer == layer) {
            uint64_t key_hash = ((uint64_t)sets[i].fusemask1) << 32 | sets[i].fusemask2;
            fuse_h0h1h2_t hs = d_fuse8_get_just_h0_h1_h2(key_hash, filter);
            uint8_t hsh = d_fuse_fingerprint(key_hash);

            if (i == hs.h0) {
                hsh ^= filter->fingerprints[hs.h1] ^ filter->fingerprints[hs.h2];
            }
            else if (i == hs.h1) {
                hsh ^= filter->fingerprints[hs.h0] ^ filter->fingerprints[hs.h2];
            }
            else {
                hsh ^= filter->fingerprints[hs.h0] ^ filter->fingerprints[hs.h1];
            }
            filter->fingerprints[i] = hsh;
        }
    }
}

// allocate enough capacity for a set containing up to 'size' elements
// caller is responsible to call fuse8_free(filter)
static inline bool fuse8_allocate(uint32_t size, fuse8_t* filter) {
    size_t capacity = 1.0 / 0.879 * size;
    capacity = capacity / FUSE_SLOTS * FUSE_SLOTS;
    cudaMallocManaged(&filter->fingerprints, capacity * sizeof(uint8_t));
    if (filter->fingerprints != NULL) {
        filter->segmentLength = capacity / FUSE_SLOTS;
        return true;
    }
    else {
        return false;
    }
}

bool fuse8_populate(const uint64_t* keys, uint32_t size, fuse8_t* filter) {
    uint64_t rng_counter = 1;
    filter->seed = fuse_rng_splitmix64(&rng_counter);
    size_t arrayLength = filter->segmentLength * FUSE_SLOTS; // size of the backing array
    fuse_fuseset_t* sets;
    cudaError_t errSets = cudaMallocManaged(&sets, arrayLength * sizeof(fuse_fuseset_t));

    if (errSets != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(errSets));
        return false;
    }

    size_t* layer;
    cudaMallocManaged(&layer, sizeof(size_t));
    *layer = 0;

    // parallelism config
    int blockSize = 128;
    int numBlocks = (filter->segmentLength + blockSize - 1) / blockSize;

    // clock_t t;
    for (int loop = 0; true; ++loop) {
        if (loop + 1 > XOR_MAX_ITERATIONS) {
            fprintf(stderr, "Too many iterations. Are all your keys unique?");
            cudaFree(sets);
            return false;
        }

        memset(sets, 0, sizeof(fuse_fuseset_t) * arrayLength);

        // t = clock();
        insert_keys << <numBlocks, blockSize >> > (keys, size, sets, filter);
        cudaDeviceSynchronize();
        // t = clock() - t;
        // double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
        // printf("### It took %f seconds to insert %zu values. \n", time_taken, size);

        fuse_fuseset_t* sets0, * sets1, * sets2;

        // t = clock();
        // copy out sets into 3 smaller arrays
        cudaError_t errSets0 = cudaMallocManaged(&sets0, filter->segmentLength * (FUSE_SLOTS / FUSE_ARITY) * sizeof(fuse_fuseset_t));
        cudaError_t errSets1 = cudaMallocManaged(&sets1, filter->segmentLength * (FUSE_SLOTS / FUSE_ARITY) * sizeof(fuse_fuseset_t));
        cudaError_t errSets2 = cudaMallocManaged(&sets2, filter->segmentLength * (FUSE_SLOTS / FUSE_ARITY) * sizeof(fuse_fuseset_t));

        for (size_t seg = 0; seg < (FUSE_SLOTS / FUSE_ARITY); seg++) {
            uint32_t dst0_offset = seg * filter->segmentLength;
            uint32_t src0_offset = seg * FUSE_ARITY * filter->segmentLength;
            memcpy(&sets0[dst0_offset], &sets[src0_offset], filter->segmentLength * sizeof(fuse_fuseset_t));
            uint32_t dst1_offset = seg * filter->segmentLength;
            uint32_t src1_offset = (seg * FUSE_ARITY + 1) * filter->segmentLength;
            memcpy(&sets1[dst1_offset], &sets[src1_offset], filter->segmentLength * sizeof(fuse_fuseset_t));
            uint32_t dst2_offset = seg * filter->segmentLength;
            uint32_t src2_offset = (seg * FUSE_ARITY + 2) * filter->segmentLength;
            memcpy(&sets2[dst2_offset], &sets[src2_offset], filter->segmentLength * sizeof(fuse_fuseset_t));
        }

        // t = clock() - t;
        // time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
        // printf("### It took %f seconds to copy subarrays. \n", time_taken);

        bool* pureCell;
        cudaMallocManaged(&pureCell, sizeof(bool));
        *pureCell = false;
        *layer = 1;
        size_t old_layer = 0;

        // size_t stack_size = 0;
        // t = clock();
        while (old_layer != *layer) {
            old_layer = *layer;

            peel_set0 << <numBlocks, blockSize >> > (sets, sets0, sets1, sets2, filter, layer, pureCell);
            cudaDeviceSynchronize();

            if (*pureCell == true) {
                *layer = *layer + 1;
                *pureCell = false;
            }

            peel_set1 << <numBlocks, blockSize >> > (sets, sets0, sets1, sets2, filter, layer, pureCell);
            cudaDeviceSynchronize();

            if (*pureCell == true) {
                *layer = *layer + 1;
                *pureCell = false;
            }

            peel_set2 << <numBlocks, blockSize >> > (sets, sets0, sets1, sets2, filter, layer, pureCell);
            cudaDeviceSynchronize();

            if (*pureCell == true) {
                *layer = *layer + 1;
                *pureCell = false;
            }

        }
        // t = clock() - t;
        // time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
        // printf("### It took %f seconds to peel over %zu values. \n", time_taken, size);

        cudaFree(pureCell);
        cudaFree(sets0);
        cudaFree(sets1);
        cudaFree(sets2);
        size_t recover_cnt = 0;
        for (int i = 0; i < arrayLength; i++) {
            if (sets[i].layer > 0) {
                recover_cnt++;
            }
        }

        if (recover_cnt == size) {
            // success
            break;
        }

        printf("peel failure, recover_cnt=%zu\n", recover_cnt);
        fflush(stdout);
        filter->seed = fuse_rng_splitmix64(&rng_counter);
    }
    
    // t = clock();
    size_t layer_size = *layer;
    while (layer_size > 0) {
        assign << <numBlocks, blockSize >> > (filter, sets, layer_size, arrayLength);
        layer_size--;
    }
    cudaDeviceSynchronize();
    // t = clock() - t;
    // double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
    // printf("### It took %f seconds to assign over %zu values. \n", time_taken, size);
    cudaFree(layer);
    cudaFree(sets);

    return true;
}

static inline bool fuse8_contain(uint64_t key, const fuse8_t* filter) {
    uint64_t hash = fuse_mix_split(key, filter->seed);
    uint8_t f = fuse_fingerprint(hash);
    uint32_t r0 = (uint32_t)hash;
    uint32_t r1 = (uint32_t)fuse_rotl64(hash, 21);
    uint32_t r2 = (uint32_t)fuse_rotl64(hash, 42);
    uint32_t r3 = (0xBF58476D1CE4E5B9 * hash) >> 32;
    uint32_t seg = fuse_reduce(r0, FUSE_SEGMENT_COUNT);
    uint32_t h0 = (seg + 0) * filter->segmentLength + fuse_reduce(r1, filter->segmentLength);
    uint32_t h1 = (seg + 1) * filter->segmentLength + fuse_reduce(r2, filter->segmentLength);
    uint32_t h2 = (seg + 2) * filter->segmentLength + fuse_reduce(r3, filter->segmentLength);
    return f == (filter->fingerprints[h0] ^ filter->fingerprints[h1] ^
        filter->fingerprints[h2]);
}

// release memory
static inline void fuse8_free(fuse8_t* filter) {
    cudaFree(filter->fingerprints);
    filter->fingerprints = NULL;
    filter->segmentLength = 0;
    cudaFree(filter);
}

bool testfuse8() {
    printf("testing fuse8\n");

    fuse8_t* filter;
    cudaMallocManaged(&filter, sizeof(fuse8_t));

    size_t size = 300000;
    fuse8_allocate(size, filter);
    // we need some set of values
    uint64_t* big_set;
    cudaMallocManaged(&big_set, size * sizeof(uint64_t));

    for (size_t i = 0; i < size; i++) {
        big_set[i] = i; // we use contiguous values
    }
    // we construct the filter
    fuse8_populate(big_set, size, filter);
    for (size_t i = 0; i < size; i++) {
        if (!fuse8_contain(big_set[i], filter)) {
            printf("bug!\n");
            return false;
        }
    }

    size_t random_matches = 0;
    size_t trials = 10000000; //(uint64_t)rand() << 32 + rand()
    for (size_t i = 0; i < trials; i++) {
        uint64_t random_key = ((uint64_t)rand() << 32) + rand();
        if (fuse8_contain(random_key, filter)) {
            if (random_key >= size) {
                random_matches++;
            }
        }
    }
    printf("fpp %3.10f (estimated) \n", random_matches * 1.0 / trials);
    printf("bits per entry %3.1f\n", fuse8_size_in_bytes(filter) * 8.0 / size);
    fuse8_free(filter);
    cudaFree(big_set);
    return true;
}

bool testfuse8(size_t size) {
    printf("testing fuse8 ");
    printf("size = %zu \n", size);

    fuse8_t* filter;
    cudaMallocManaged(&filter, sizeof(fuse8_t));

    fuse8_allocate(size, filter);
    // we need some set of values
    uint64_t* big_set;
    cudaMallocManaged(&big_set, size * sizeof(uint64_t));

    for (size_t i = 0; i < size; i++) {
        big_set[i] = i; // we use contiguous values
    }
    // we construct the filter
    fuse8_populate(big_set, size, filter); // warm the cache
    for (size_t times = 0; times < 1; times++) {
        clock_t t;
        t = clock();
        fuse8_populate(big_set, size, filter);
        t = clock() - t;
        double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
        printf("It took %f seconds to build an index over %zu values. \n",
            time_taken, size);
    }
    fuse8_free(filter);
    cudaFree(big_set);
    return true;
}

int main() {
    for (size_t s = 10000000; s <= 10000000; s += 10000000) {
        // testfuse8(s);
        testfuse8();

        printf("\n");
    }
}