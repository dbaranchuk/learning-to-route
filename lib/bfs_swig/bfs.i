%module bfs

%{
    #define SWIG_FILE_WITH_INIT
    #include "bfs.h"
%}

%include "numpy.i"
%include "typemaps.i"

%init %{
    import_array();
%}

%apply (int DIM1, int DIM2, int *IN_ARRAY2) {(int maxelements, int MaxM, int *edges)}
%apply (int DIM1, int* IN_ARRAY1) {(int k, int *initial_vertex_ids),
                                   (int m, int *gts)}
%apply (int DIM1, int DIM2, int *IN_ARRAY2) {(int nq, int max_path_length, int *visited_ids)}
%apply (int DIM1, int DIM2, int* INPLACE_ARRAY2) {(int d1,  int d2, int *distances)}
%apply int *INPUT {int *margin, int *nt}

%include "bfs.h"
