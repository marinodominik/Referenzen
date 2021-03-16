include ../../../../.Makefile.global.variables

CXXINCLUDE = -I../../../../lib/$(SRCDIRNAME) -I../../basics -I.. -I../skills -I. -I../../policy
CXXFLAGS   = -fno-stack-protector

ASSEMBLY_DIR=""
ASSEMBLY_NAME=all

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME): PRINT_ENTER_DIRECTORY $(OBJS)
	$(PRINT_LEAVE_DIRECTORY)

include ../../../../.Makefile.global.rules
