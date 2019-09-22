%module pySemanticFusionInterface
%include <std_vector.i>
%include <std_shared_ptr.i>
%{
    #define SWIG_FILE_WITH_INIT
    #include <caffe/caffe.hpp>
    #include <semantic_fusion/SemanticFusionInterface.h>
%}
%shared_ptr( caffe::Blob<float> )
//%template(PtrFloatBlob) shared_ptr( caffe::Blob<float> );


%include <numpy.i>

%init %{
import_array();
%}

%numpy_typemaps(unsigned short             , NPY_USHORT    , int)
%numpy_typemaps(unsigned char             , NPY_UBYTE    , int)
%apply (unsigned char* IN_ARRAY1, int DIM1){(ImagePtr rgb_arr, int n_rgb)};
%apply (unsigned short* IN_ARRAY1, int DIM1){(DepthPtr depth_arr, int n_depth)};
%apply (float** ARGOUTVIEWM_FARRAY4, int *DIM1, int *DIM2, int *DIM3, int *DIM4) {(float** probs, int* ps1, int* ps2, int* ps3, int* ps4)}

%include semantic_fusion/SemanticFusionInterface.h


