%module pyCaffeInterface
%include <std_string.i>
%include <std_shared_ptr.i>

%{
#define SWIG_FILE_WITH_INIT
#include <cnn_interface/CaffeInterface.h>
%}
%shared_ptr( caffe::Blob<float> )

%include "../numpy.i"

%init %{
import_array();
%}

%numpy_typemaps(unsigned short             , NPY_USHORT    , int)
%numpy_typemaps(unsigned char             , NPY_UBYTE    , int)
%apply (unsigned char* IN_ARRAY1, int DIM1){(ImagePtr rgb_arr, int n_rgb)};
%apply (unsigned short* IN_ARRAY1, int DIM1){(DepthPtr depth_arr, int n_depth)};
%include cnn_interface/CaffeInterface.h

