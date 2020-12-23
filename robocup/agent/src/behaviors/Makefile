include ../../../.Makefile.global.variables

CXXINCLUDE	= -I../../../lib/$(SRCDIRNAME) -I.. -I../basics -I../policy -I. -Iskills -Iview
CXXFLAGS        = -fno-stack-protector

ASSEMBLY_DIR=$(ACTOBJDIR)/
ASSEMBLY_NAME=libbehaviors.a

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME): PRINT_ENTER_DIRECTORY $(OBJS) MAKE_SUBDIRECTORYS
	$(PRINT_LEAVE_DIRECTORY)

include ../../../.Makefile.global.rules
