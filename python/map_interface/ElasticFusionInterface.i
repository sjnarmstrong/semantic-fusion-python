#ifndef WIN32
#  define EFUSION_API
#else
#  ifdef efusion_EXPORTS
#    define EFUSION_API __declspec(dllexport)
#  else
#    define EFUSION_API __declspec(dllimport)
#  endif
#endif

%module pyElasticFusionInterface
%include <std_vector.i>
%include <std_string.i>
%{
    #define SWIG_FILE_WITH_INIT
    #include <map_interface/ElasticFusionInterface.h>
%}



%include "../numpy.i"

%init %{
import_array();
%}


%numpy_typemaps(unsigned short             , NPY_USHORT    , int)
%numpy_typemaps(unsigned char             , NPY_UBYTE    , int)
%apply (unsigned char* IN_ARRAY1, int DIM1){(ImagePtr rgb_arr, int n_rgb)};
%apply (unsigned short* IN_ARRAY1, int DIM1){(DepthPtr depth_arr, int n_depth)};

%include Utils/Intrinsics.h
%include Utils/Resolution.h
%include map_interface/ElasticFusionInterface.h