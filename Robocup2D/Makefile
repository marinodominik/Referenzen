include .Makefile.global.variables

DIRS=	lib   \
	agent \
	coach

ASSEMBLY_DIR=""
ASSEMBLY_NAME=all

$(ASSEMBLY_DIR)$(ASSEMBLY_NAME):
ifeq ($(PLAIN_MODE), 0)
	@$(foreach DIR, $(DIRS), $(MAKE) -s -C $(DIR) $(MAKECMDGOALS) && ) $(PRINT_ALLDONE)
else
	@$(foreach DIR, $(DIRS), $(MAKE) -C $(DIR) $(MAKECMDGOALS) && ) $(PRINT_ALLDONE)
endif

include .Makefile.global.rules
