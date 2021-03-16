include ../../.Makefile.global.variables

CXXINCLUDE = \
-I../../lib/$(SRCDIRNAME) \
-I. \
-Ibasics \
-Ibehaviors \
-Ibehaviors/skills \
-Ibehaviors/view \
-Ipolicy

CXXLIBS    = -L../../lib/$(OBJDIRNAME) -lbs

CXXFLAGS   = -Wall -O2 -fno-stack-protector -pipe

ASSEMBLY_DIR=$(ACTBINDIR)/
ASSEMBLY_NAME=FRA-UNIted_Agent

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME): PRINT_ENTER_DIRECTORY $(OBJS) MAKE_SUBDIRECTORYS
	$(PRINT_LINKTARGET)
ifeq ($(PLAIN_MODE), 0)
	@$(CXX) -o $@ $(shell find ../obj/ -name "*.o" -exec printf "%s " {} \;) $(CXXLIBS)
else
	$(CXX) -o $@ $(shell find ../obj/ -name "*.o" -exec printf "%s " {} \;) $(CXXLIBS)
endif
	$(PRINT_LINKTARGETDONE)

	$(PRINT_LEAVE_DIRECTORY)

include ../../.Makefile.global.rules
