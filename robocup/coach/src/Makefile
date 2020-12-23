include ../../.Makefile.global.variables

CXXINCLUDE = -I. -I../../lib/$(SRCDIRNAME)
CXXLIBS    = -L../../lib/$(OBJDIRNAME) -lbs
CXXFLAGS   = -fno-stack-protector -Wall -pipe -O2

ASSEMBLY_DIR=$(ACTBINDIR)/
ASSEMBLY_NAME=FRA-UNIted_Coach

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME): PRINT_ENTER_DIRECTORY $(OBJS) MAKE_SUBDIRECTORYS
	$(PRINT_LINKTARGET)
ifeq ($(PLAIN_MODE), 0)
	@$(CXX) -o $@ $(shell find ../obj/ -name "*.o" -printf "%p ") $(CXXLIBS)
else
	$(CXX) -o $@ $(shell find ../obj/ -name "*.o" -printf "%p ") $(CXXLIBS)
endif
	$(PRINT_LINKTARGETDONE)

	$(PRINT_LEAVE_DIRECTORY)

include ../../.Makefile.global.rules
