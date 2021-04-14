# - Find LIBGP
# Find the native FFTW includes and library
#
#  FFTW_INCLUDES    - where to find fftw3.h
#  FFTW_LIBRARIES   - List of libraries when using FFTW.
#  FFTW_FOUND       - True if FFTW found.

if (LIBGP_INCLUDES)
    # Already in cache, be silent
    set(LIBGP_FIND_QUIETLY TRUE)
endif (LIBGP_INCLUDES)
set(LIBGP_INCLUDES "/home/twelsche/libgp")
find_path(LIBGP_INCLUDES gp.h)

find_library(LIBGP_LIBRARIES NAMES libgp)

# handle the QUIETLY and REQUIRED arguments and set LIBGP_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBGP DEFAULT_MSG LIBGP_LIBRARIES LIBGP_INCLUDES)

mark_as_advanced(LIBGP_LIBRARIES LIBGP_INCLUDES)
