include ../../../.Makefile.global.variables

CXXINCLUDE = -I../../../lib/$(SRCDIRNAME) -I.. -I. -I../behaviors -I../behaviors/skills -I../behaviors/view -I../policy
CXXFLAGS   = -fno-stack-protector

ASSEMBLY_DIR=$(ACTOBJDIR)/
ASSEMBLY_NAME=libbasics.a

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME): PRINT_ENTER_DIRECTORY $(OBJS)
	$(PRINT_LEAVE_DIRECTORY)

include ../../../.Makefile.global.rules
