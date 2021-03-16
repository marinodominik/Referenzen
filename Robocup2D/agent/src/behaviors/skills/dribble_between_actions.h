#ifndef _DRIBBLE_AROUND_ACTIONS_H_
#define _DRIBBLE_AROUND_ACTIONS_H_
/* Brrr: Nasty CPP makros to prevent double declaration */
#define ACTION_NAMES       \
	C(DA_DASHING)            \
	C(DA_TURNING)            \
	C(DA_TACKLING)           \
	C(DA_COLLISION_KICK)     \
  C(DA_KICK_HE_ISNT_AT)    \
  C(DA_KICK_HE_ISNT_MOVING_AT)    \
  C(DA_KICK_MY_DIR)        \
  C(DA_KICK_TOWARDS_GOAL)  \
  C(DA_KICK_KEEP)          \
  C(DA_KICK_SAFEST)        \
  C(DA_KICK_OPP_LOOKS)     \
  C(DA_KICK_OPP_MOVES)     \
  C(DA_KICK_FALLBACK)      \
  C(DA_KICK_AHEAD)         \
	C(DA_NO_ACTION)
#define C(X) X,
enum DribbleAction { ACTION_NAMES DA_TOP  };
#undef C
#define C(X) #X,
const char * const dribbleActionNames[] = {ACTION_NAMES };
#endif
