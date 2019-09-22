%module pyLogReader
%include <std_string.i>
%{
    #define SWIG_FILE_WITH_INIT
    #include <utilities/LogReader.h>
    #include <utilities/PNGLogReader.h>
    #include <utilities/RawLogReader.h>
    #include <utilities/LiveLogReader.h>

%}


%include "../numpy.i"

%init %{
    import_array();
%}

%numpy_typemaps(unsigned char     , NPY_UBYTE    , int)
%numpy_typemaps(unsigned short             , NPY_USHORT    , int)
%apply (unsigned char* INPLACE_ARRAY1, int DIM1) {(unsigned char* out_arr, int n)};
%apply (unsigned short* INPLACE_ARRAY1, int DIM1) {(unsigned short* out_arr, int n)};
%include utilities/LogReader.h

//%apply (unsigned char* INPLACE_ARRAY1, int DIM1) {(unsigned char *rgb_arr, int n)}

//%extend LogReader{
//    void updateRGBVector(double* seq, int n)
//    {
//        assert ( n==$self->getImageSize() );
//        memcpy ( $self->rgb, seq, n );
//    }
//}




%include utilities/PNGLogReader.h
%include utilities/RawLogReader.h
%include utilities/LiveLogReader.h