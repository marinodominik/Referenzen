#ifndef AGENT_SRC_PLAYERFLAGSET_H_
#define AGENT_SRC_PLAYERFLAGSET_H_

#include "basics/globaldef.h"

class PlayerFlagSet
{
    bool players[ 2 ][ NUM_PLAYERS + 1 ];
public:
    PlayerFlagSet();
    void set_all();
    void unset_all();
    bool get( int team, int num ) const;
    void set( int team, int num );
    void unset( int team, int num );
};

#endif /* AGENT_SRC_PLAYERFLAGSET_H_ */
