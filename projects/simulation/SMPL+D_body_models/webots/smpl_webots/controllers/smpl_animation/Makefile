# Webots Makefile system
#
# You may add some variable definitions hereafter to customize the build process
# See documentation in $(WEBOTS_HOME_PATH)/resources/Makefile.include

space :=
space +=
WEBOTS_HOME_PATH=$(subst $(space),\ ,$(strip $(subst \,/,$(WEBOTS_HOME))))
ifndef WEBOTS_SKIN_ANIMATION_PATH
WEBOTS_SKIN_ANIMATION_PATH = ../../libraries
endif
INCLUDE = -I"$(WEBOTS_SKIN_ANIMATION_PATH)/smpl_util/include"
LIBRARIES = -L"$(WEBOTS_SKIN_ANIMATION_PATH)/smpl_util" -lsmpl_util 
# Do not modify the following: this includes Webots global Makefile.include
include /media/charsyme/HDD_drive/Linux/webots_development/webots/resources/Makefile.include

