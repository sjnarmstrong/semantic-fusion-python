%module pyTypes
%include <std_string.i>
%{
    #include <utilities/Types.h>
    #include <vector>
%}
%include utilities/Types.h
%include "std_vector.i"
namespace std {
        %template(VectorOfClassColour) vector<ClassColour>;
}
