#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gloo" for configuration "Release"
set_property(TARGET gloo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gloo PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/pytorch/torch/lib/libgloo.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS gloo )
list(APPEND _IMPORT_CHECK_FILES_FOR_gloo "/pytorch/torch/lib/libgloo.a" )

# Import target "gloo_cuda" for configuration "Release"
set_property(TARGET gloo_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gloo_cuda PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/pytorch/torch/lib/libgloo_cuda.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS gloo_cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_gloo_cuda "/pytorch/torch/lib/libgloo_cuda.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
