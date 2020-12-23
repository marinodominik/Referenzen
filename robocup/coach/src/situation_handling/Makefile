include ../../../.Makefile.global.variables

CXXINCLUDE = -I. -I.. -I../../../lib/$(SRCDIRNAME)
CXXFLAGS   = -Wall -fno-stack-protector

ASSEMBLY_DIR=$(ACTOBJDIR)/
ASSEMBLY_NAME=libsituation_handling.a

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME): PRINT_ENTER_DIRECTORY $(OBJS)
	$(PRINT_LEAVE_DIRECTORY)

include ../../../.Makefile.global.rules
