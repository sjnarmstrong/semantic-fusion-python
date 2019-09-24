find_package(SWIG REQUIRED)
find_package ( PythonInterp REQUIRED )
find_package(PythonLibs REQUIRED)
find_package ( NumPy REQUIRED )
INCLUDE(${SWIG_USE_FILE})
include_directories(${PYTHON_INCLUDE_DIRS})

#add_subdirectory(semantic_fusion)
if(NOT DEFINED PYTHON_MODULE_PATH)

    execute_process ( COMMAND ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print( sysconfig.get_python_lib( plat_specific=True, prefix='${CMAKE_INSTALL_PREFIX}' ) )"
            OUTPUT_VARIABLE _ABS_PYTHON_MODULE_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE )
    #execute_process ( COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    #        OUTPUT_VARIABLE _ABS_PYTHON_MODULE_PATH
    #        OUTPUT_STRIP_TRAILING_WHITESPACE )

    get_filename_component ( _ABS_PYTHON_MODULE_PATH ${_ABS_PYTHON_MODULE_PATH} ABSOLUTE )
    file ( RELATIVE_PATH _REL_PYTHON_MODULE_PATH ${CMAKE_INSTALL_PREFIX} ${_ABS_PYTHON_MODULE_PATH} )

    set ( PYTHON_MODULE_PATH
         ${_REL_PYTHON_MODULE_PATH}
       )

endif(NOT DEFINED PYTHON_MODULE_PATH)

message("Setting module install path to ${PYTHON_MODULE_PATH}")

function(add_swig_item swig_file_name swig_folder_name)
    set_property(SOURCE ${swig_folder_name}/${swig_file_name}.i PROPERTY CPLUSPLUS ON)
#    set_property(SOURCE ${swig_folder_name}/${swig_file_name}.i PROPERTY SWIG_FLAGS "-includeall")
    set_property(SOURCE ${swig_folder_name}/${swig_file_name}.i PROPERTY SWIG_MODULE_NAME py${swig_file_name})
    swig_add_library(py${swig_file_name}
            LANGUAGE python
            SOURCES ${swig_folder_name}/${swig_file_name}.i
            )

    set_property(TARGET py${swig_file_name} PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

    target_include_directories(py${swig_file_name} PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            ${Python_INCLUDE_DIR}
            ${NUMPY_INCLUDES}
            )

    target_link_libraries(py${swig_file_name}
            ${CMAKE_PROJECT_NAME}
            ${PYTHON_LIBRARIES}
            ${NUMPY_LIBRARIES}
            )
    INSTALL ( FILES ${CMAKE_CURRENT_BINARY_DIR}/_py${swig_file_name}.so DESTINATION ${PYTHON_MODULE_PATH}/${CMAKE_PROJECT_NAME}/${swig_folder_name} )
    INSTALL ( FILES ${CMAKE_CURRENT_BINARY_DIR}/py${swig_file_name}.py DESTINATION ${PYTHON_MODULE_PATH}/${CMAKE_PROJECT_NAME}/${swig_folder_name} )
endfunction(add_swig_item)



INSTALL ( FILES __init__.py.in DESTINATION ${PYTHON_MODULE_PATH}/${CMAKE_PROJECT_NAME} RENAME __init__.py )

INSTALL ( FILES __init__.py.in DESTINATION ${PYTHON_MODULE_PATH}/${CMAKE_PROJECT_NAME}/semantic_fusion RENAME __init__.py )
add_swig_item(SemanticFusionCuda semantic_fusion)
add_swig_item(SemanticFusionInterface semantic_fusion)

INSTALL ( FILES __init__.py.in DESTINATION ${PYTHON_MODULE_PATH}/${CMAKE_PROJECT_NAME}/map_interface RENAME __init__.py )
add_swig_item(ElasticFusionInterface map_interface)
add_swig_item(ElasticFusionCuda map_interface)

INSTALL ( FILES __init__.py.in DESTINATION ${PYTHON_MODULE_PATH}/${CMAKE_PROJECT_NAME}/cnn_interface RENAME __init__.py )
add_swig_item(CaffeInterface cnn_interface)

INSTALL ( FILES __init__.py.in DESTINATION ${PYTHON_MODULE_PATH}/${CMAKE_PROJECT_NAME}/gui RENAME __init__.py )
add_swig_item(Gui gui)
add_swig_item(GuiCuda gui)

INSTALL ( FILES __init__.py.in DESTINATION ${PYTHON_MODULE_PATH}/${CMAKE_PROJECT_NAME}/utilities RENAME __init__.py )
add_swig_item(Array utilities)
add_swig_item(JPEGLoader utilities)
#add_swig_item(LiveLogReader utilities)
add_swig_item(LogReader utilities)
#add_swig_item(OpenNI2Interface utilities)
#add_swig_item(PNGLogReader utilities)
#add_swig_item(RawLogReader utilities)
add_swig_item(Stopwatch utilities)
add_swig_item(ThreadMutexObject utilities)
add_swig_item(Types utilities)
