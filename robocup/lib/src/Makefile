include ../../.Makefile.global.variables

CXXFLAGS=-Wall -O2 -pipe

ASSEMBLY_DIR=$(ACTOBJDIR)/
ASSEMBLY_NAME=libbs.a

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME): PRINT_ENTER_DIRECTORY $(OBJS)
	$(PRINT_PACKLIB)
ifeq ($(PLAIN_MODE), 0)
	@ar rc  $@ $(OBJS)
	@ranlib $@
else
	ar rc  $@ $(OBJS)
	ranlib $@
endif
	$(PRINT_PACKLIBDONE)

	$(PRINT_LEAVE_DIRECTORY)

include ../../.Makefile.global.rules
