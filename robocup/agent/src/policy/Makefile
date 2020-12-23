include ../../../.Makefile.global.variables

CXXINCLUDE = -I../../../lib/$(SRCDIRNAME) -I.. -I../basics -I../behaviors -I../behaviors/skills -I../behaviors/view -I.
CXXFLAGS   = -fno-stack-protector

ASSEMBLY_DIR=$(ACTOBJDIR)/
ASSEMBLY_NAME=libpolicy.a

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME): PRINT_ENTER_DIRECTORY $(OBJS)
	$(PRINT_LEAVE_DIRECTORY)

include ../../../.Makefile.global.rules
