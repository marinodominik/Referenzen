#include "PlayerFlagSet.h"

PlayerFlagSet::PlayerFlagSet()
{
    unset_all();
}

void PlayerFlagSet::set_all()
{
    for( int t = 0; t < 2; t++ )
        for( int i = 0; i < NUM_PLAYERS + 1; i++ )
            players[ t ][ i ] = true;
}

void PlayerFlagSet::unset_all()
{
    for( int t = 0; t < 2; t++ )
        for( int i = 0; i < NUM_PLAYERS + 1; i++ )
            players[ t ][ i ] = false;
}

bool PlayerFlagSet::get( int team, int num ) const
{
    return players[ team ][ num ];
}

void PlayerFlagSet::set( int team, int num )
{
    players[ team ][ num ] = true;
}

void PlayerFlagSet::unset( int team, int num )
{
    players[ team ][ num ] = false;
}
