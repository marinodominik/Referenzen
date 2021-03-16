include ../../../../.Makefile.global.variables

CXXINCLUDE = -I../../../../lib/$(SRCDIRNAME) -I../../ -I../../basics -I../../policy -I.. -I../view -I.
CXXFLAGS   = -fno-stack-protector

ASSEMBLY_DIR=""
ASSEMBLY_NAME=all

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME): PRINT_ENTER_DIRECTORY $(OBJS)
	$(PRINT_LEAVE_DIRECTORY)

include ../../../../.Makefile.global.rules
