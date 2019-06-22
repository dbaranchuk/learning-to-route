#include <vector>
#include <cstdio>
#include <queue>
#include <iostream>
#include <assert.h>
#include <omp.h>
#include <unordered_set>

typedef unsigned idx_t;

struct Vertex{
    idx_t vertex_id;
    std::vector<idx_t> prev_vertex_ids;

    bool is_visited = false;
    size_t min_path_length = 0;
};

void bfs(int maxelements, int MaxM, int *edges,  // matrix [maxelements, MaxM]
         int k, int *initial_vertex_ids,         // vector [n_queries]
         int m, int *gts,                        // vector [n_queries]
         int d1, int d2, int *distances,         // matrix [n_queries, maxelements]
         int *margin,                             // number
         int *nt);                                // number

void bfs_visited_ids(int maxelements, int MaxM, int *edges,              // matrix [maxelements, MaxM]
                     int m, int *gts,                                    // vector [n_queries]
                     int d1, int d2, int *distances,                     // matrix [n_queries, max_path_length]
                     int nq, int max_path_length, int *visited_ids,      // matrix [n_queries, max_path_length]
                     int *nt);                                           // number