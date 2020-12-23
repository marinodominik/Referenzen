#ifndef _PLANNING2_H
#define _PLANNING2_H

#include "PlayerSet.h"
#include "globaldef.h"
#include "Vector.h"
#include "abstract_mdp.h"
#include "planning.h"
#include "log_macros.h"
#include "options.h"
#include "valueparser.h"
#include "tools.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

class Planning2 {
 private:
  static int generate_pactions(const AState &state, AAction *actions);

 public:
  static Vector compute_potential_pos(const AState & state);   // compute potential of that state
};

#endif
