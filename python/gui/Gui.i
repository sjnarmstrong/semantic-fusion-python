%module pyGui
%include <std_vector.i>
%include <std_string.i>

%{
//#include <pangolin/display/display.h>
#include <gui/Gui.h>
%}
%include gui/Gui.h
%extend Gui{
        bool ShouldQuit() {
            return pangolin::ShouldQuit();
        }
}