#include "ws_info.h"

#include "ws.h"
#include "ws_memory.h"
#include "intercept.h"
#include <string.h>

#include "PlayerSet.h"
#include "options.h"
#include "tools.h"
#include "log_macros.h" //test
#include "macro_msg.h"

char WSinfo::cvCurrentOpponentIdentifier = '_';

Cmd *WSinfo::current_cmd      = 0;
Cmd *WSinfo::last_cmd         = 0;

WS const* WSinfo::ws          = 0;
WS const* WSinfo::ws_full     = 0;

PPlayer WSinfo::me            = 0;
PPlayer WSinfo::me_full       = 0;

PPlayer WSinfo::his_goalie    = 0;

Ball const* WSinfo::ball      = 0;
Ball const* WSinfo::ball_full = 0;

PlayerSet WSinfo::alive_teammates;
PlayerSet WSinfo::alive_teammates_without_me;
PlayerSet WSinfo::alive_opponents;

PlayerSet WSinfo::valid_teammates;
PlayerSet WSinfo::valid_teammates_without_me;
PlayerSet WSinfo::valid_opponents;

//JK PASS_MSG_HACK begin
bool WSinfo::jk_pass_msg_set = false;
bool WSinfo::jk_pass_msg_rec = false;
char WSinfo::jk_pass_msg[80];
long WSinfo::jk_pass_msg_rec_time = -99;
float WSinfo::jk_pass_msg_x;
float WSinfo::jk_pass_msg_y;
//JK PASS_MSG_HACK end

int WSinfo::relevant_teammate[11];
int WSinfo::num_relevant_teammates;



PlayerSet WSinfo::pset_tmp;

double  WSinfo::my_team_pos_of_offside_line_cache;
bool    WSinfo::my_team_pos_of_offside_line_cache_ok;
double  WSinfo::his_team_pos_of_offside_line_cache;
bool    WSinfo::his_team_pos_of_offside_line_cache_ok;
PPlayer WSinfo::teammate_with_newest_pass_info;
bool    WSinfo::teammate_with_newest_pass_info_ok;

PPlayer WSinfo::numbered_valid_players[ 2 * NUM_PLAYERS + 1 ];
WSinfo::PlayerInactivities WSinfo::player_inactivities;

////////////////////////////////////////////////////////////////////////////////

WSinfo::PlayerInactivities::PlayerInactivities()
{
    for( int i = 0; i < 2 * NUM_PLAYERS; i++ )
        validAtTime[ i ] = -1;
}

bool WSinfo::init( const WS *worldstate, const WS *worldstate_full )
{
    ws            = worldstate;
    ws_full       = worldstate_full;

    me         = 0;
    me_full    = 0;

    his_goalie = 0;

    ball       = &( ws->ball );
    ball_full  = 0;

    alive_teammates.num            = 0;
    alive_teammates_without_me.num = 0;
    alive_opponents.num            = 0;

    valid_teammates.num            = 0;
    valid_teammates_without_me.num = 0;
    valid_opponents.num            = 0;

    my_team_pos_of_offside_line_cache_ok  = false;
    his_team_pos_of_offside_line_cache_ok = false;
    teammate_with_newest_pass_info_ok     = false;

    for( int i = 0; i < 2 * NUM_PLAYERS; i++ )
    {
        numbered_valid_players[ i ] = 0;
    }

    ////////////////////////////////////////////////////////////////

    int my_number = ClientOptions::player_no;

    set_relevant_teammates_default();

    ////////////////////////////////////////////////////////////////

    for( int i = 0; i < ws->my_team_num; i++ )
    {
        if( ws->my_team[ i ].alive && ws->my_team[ i ].number == my_number )
        {
            me = &( ws->my_team[ i ] );
            break;
        }
    }
    if( me == 0 )
    {
        ERROR_OUT << "\n-------------\nme was not set!\n" << *ws << "\n---------------";
    }

    if( ws_full )
    {
        for( int i = 0; i < ws_full->my_team_num; i++ )
        {
            if( ws_full->my_team[ i ].alive && ws_full->my_team[ i ].number == my_number )
            {
                me_full = &( ws_full->my_team[ i ] );
                break;
            }
        }
        ball_full = &(ws_full->ball);
    }

    ////////////////////////////////////////////////////////////////

    for( int i = 0; i < ws->my_team_num; i++ )
    {
        if( ws->my_team[ i ].alive )
        {
            alive_teammates.p[ alive_teammates.num ]= &( ws->my_team[ i ] );
            alive_teammates.num++;

            if( ws->my_team[ i ].number != my_number )
            {
                alive_teammates_without_me.p[ alive_teammates_without_me.num ] = &( ws->my_team[ i ] );
                alive_teammates_without_me.num++;
            }
        }
    }

    for( int i = 0; i < ws->his_team_num; i++ )
    {
        if( ws->his_team[ i ].alive )
        {
            alive_opponents.p[ alive_opponents.num ]= &( ws->his_team[ i ] );
            alive_opponents.num++;
        }
    }

    ////////////////////////////////////////////////////////////////

    for( int i = 0; i < ws->my_team_num; i++ )
    {
        if( is_teammate_pos_valid( &ws->my_team[ i ] ) )
        {
            valid_teammates.p[ valid_teammates.num ] = &( ws->my_team[ i ] );
            valid_teammates.num++;
            if( ws->my_team[ i ].number != my_number )
            {
                valid_teammates_without_me.p[ valid_teammates_without_me.num ] = &( ws->my_team[ i ] );
                valid_teammates_without_me.num++;
            }
        }
    }

    for( int i = 0; i < ws->his_team_num; i++ )
    {
        if( is_opponent_pos_valid( &ws->his_team[ i ] ) )
        {
            valid_opponents.p[ valid_opponents.num ]= &( ws->his_team[ i ] );
            valid_opponents.num++;
        }
    }

    ////////////////////////////////////////////////////////////////

    for( int i = 0; i <  valid_teammates.num; i++ )
    {
        int number = valid_teammates[ i ]->number;
        if( number > 0 )
        {
            numbered_valid_players[ number ] = valid_teammates[ i ];
        }
    }

    for( int i = 0; i < valid_opponents.num; i++ )
    {
        int number = valid_opponents[ i ]->number;
        if( number > 0 )
        {
            numbered_valid_players[ number + NUM_PLAYERS ] = valid_opponents[ i ];
        }
    }

    if( ws->his_goalie_number > 0 )
    {
        his_goalie = get_opponent_by_number( ws->his_goalie_number );
    }

    return true;
}

bool WSinfo::is_my_pos_valid()
{
    return is_teammate_pos_valid( me );
}

bool WSinfo::is_teammate_pos_valid( const PPlayer player )
{
    return player->alive && ( ws->time - player->time ) <= 30; //30 is the value which was used in Seattle
}

bool WSinfo::is_opponent_pos_valid( const PPlayer player )
{
    if( player )
    {
        if( player->pos.distance( WSinfo::me->pos ) < ServerOptions::visible_distance - .1 && player->age > 1 )
        {
            LOG_POL( 0, "WSINFO: PLAYER should be in feel range, but age too old -> do not consider!!!" );
            return true; // TG: changed from false to true
        }
    }

    return player->alive && ( ws->time - player->time ) <= 60; //60 is the value which was used in Seattle
    //in Seattle there was some extra treatment of the opponent goalie which never
    //expired, but this should probably be done at a higher level!
}

bool WSinfo::is_ball_pos_valid()
{
    return ( ws->time - ws->ball.time ) <= 7; // 7 is the value which was used in Seattle
}

bool WSinfo::is_ball_kickable_for( const PPlayer player, const Vector &ballPos /* = WSinfo::ball->pos */ )
{
    return ( player->pos.distance( ballPos ) <= player->kick_radius );
}

bool WSinfo::is_ball_kickable( const Vector &ballPos /* = WSinfo::ball->pos */ )
{
    return is_ball_kickable_for( me, ballPos );
}

bool WSinfo::is_ball_pos_valid_and_kickable()
{
    return is_ball_pos_valid() && is_ball_kickable();
}

double WSinfo::get_tackle_probability_for( const PPlayer player, bool foul /* = false */, const Vector &ballPos /* = WSinfo::ball->pos */ )
{
    Vector playerToBall = ( ballPos - player->pos ).rotate( -player->ang.get_value() );

    double tackleDist = playerToBall.getX() > 0.0 ? ServerOptions::tackle_dist : ServerOptions::tackle_back_dist;

    double tackleExpo = ServerOptions::tackle_exponent;

    if( foul )
    {
        foul = false;

        for( int i = 0; i < valid_opponents.num; i++ )
        {
            if( is_ball_kickable_for( valid_opponents.get_player_by_number(i) ) )
            {
                foul = true;
                tackleExpo = ServerOptions::foul_exponent;
                break;
            }
        }
    }

    return fabs( tackleDist ) <= 1.0e-5 ? 0.0 : ( pow( fabs( playerToBall.getX() ) / tackleDist, tackleExpo ) + pow( fabs( playerToBall.getY() ) / ServerOptions::tackle_width, tackleExpo ) );
}

double WSinfo::get_tackle_probability( bool foul /* = false */, const Vector &ballPos /* = WSinfo::ball->pos */ )
{
    return get_tackle_probability_for( me, foul, ballPos );
}

bool WSinfo::is_player_probably_inactive_after_tackling( const PPlayer player )
{
  return (   (player->tackle_flag && player->age == 0)
          || (ws->time - player->tackle_time < ServerOptions::tackle_cycles) );
}

bool WSinfo::is_player_inactive_after_being_fouled( PPlayer player )
{
    return  ( ws->play_mode == PM_PlayOn ) &&
            ( ws->time - WSmemory::get_last_fouled_time() < ServerOptions::foul_cycles ) &&
            ( player && player->number == WSmemory::get_last_fouled_opponent_number() );
}

void WSinfo::get_player_inactivity_interval_after_tackling( const PPlayer player, int& minNoOfInactCycles, int& maxNoOfInactCycles )
{
    minNoOfInactCycles = 0;
    maxNoOfInactCycles = 0;

    if( ws->time - player->tackle_time < ServerOptions::tackle_cycles )
    {
        maxNoOfInactCycles = ServerOptions::tackle_cycles - ( ws->time - player->tackle_time );

        if( maxNoOfInactCycles < 0 ) maxNoOfInactCycles = 0;
        //LOG_POL(4,"TACKHIST Pl "<<player->number<<" @ "<<player->team<<" tackled at t="<<player->tackle_time
        //    <<" -> remaining inactivity MAXIMALLY "<<maxNoOfInactCycles);

        if( ws->time - player->action_time < ServerOptions::tackle_cycles )
        {
            minNoOfInactCycles = ServerOptions::tackle_cycles - ( ws->time - player->action_time );

            if( minNoOfInactCycles < 0 ) minNoOfInactCycles = 0;
            //LOG_POL(0,"TACKHIST Pl "<<player->number<<" @ "<<player->team<<" moved at t="<<player->action_time
            //    <<" -> remaining inactivity MINIMALLY "<<minNoOfInactCycles);
        }
    }
    //LOG_POL(4,"TACKHIST Pl "<<player->number<<" @ "<<player->team<<": Expected cycles"
    //      << " of inactivity ["<<minNoOfInactCycles<<","<<maxNoOfInactCycles<<"]");
}

void WSinfo::get_player_inactivity_interval_after_being_fouled(
        PPlayer player,
        int& minNoOfInactCycles,
        int& maxNoOfInactCycles )
{
    minNoOfInactCycles = 0;
    maxNoOfInactCycles = 0;
    // Example: I have played foul (tackled with foul flag true) in t=1234.
    //          I the tackle-foul succeeds *and* the referee does *not* detect it, then
    //          the fouled player remains inactive in 1235 ... 1239 (1235+foul_cycles-1).
    // Note: The foul has not been detected, if the play mode stays play_on.
    if( ws->play_mode == PM_PlayOn &&
        ws->time - WSmemory::get_last_fouled_time() < ServerOptions::foul_cycles )
    {
        if( player && player->number == WSmemory::get_last_fouled_opponent_number() )
        {
            minNoOfInactCycles = ServerOptions::foul_cycles - ( ws->time - WSmemory::get_last_fouled_time() - 1 );
            if( minNoOfInactCycles < 0 )
                minNoOfInactCycles = 0;
            maxNoOfInactCycles = minNoOfInactCycles;
        }
    }
    /*LOG_POL(4,"FOULHIST Pl "<<player->number<<" @ "<<player->team<<", t="<<ws->time<<": Expected cycles"
     << " of inactivity ["<<minNoOfInactCycles<<","<<maxNoOfInactCycles<<"], WSmemory::get_last_fouled_time()="
     <<WSmemory::get_last_fouled_time()<<", WSmemory::get_last_fouled_opponent_number()="<<WSmemory::get_last_fouled_opponent_number()
     <<", playmode="<<ws->play_mode<<", playon="<<PM_PlayOn<<", foul_cyc="<<ServerOptions::foul_cycles);*/
}

void WSinfo::get_player_inactivity_interval(
        PPlayer player,
        int& minNoOfInactCycles,
        int& maxNoOfInactCycles )
{
    int index = NUM_PLAYERS * ( player->team - 1 ) + player->number - 1;

    if( player && ( player->team == 1 || player->team == 2 ) )
    {
        if( player_inactivities.validAtTime[ index ] == WSinfo::ws->time )
        {
            //use cached results
            minNoOfInactCycles = player_inactivities.minInact[ index ];
            maxNoOfInactCycles = player_inactivities.maxInact[ index ];
            return;
        }
    }

    // no cache results available, compute them now!
    get_player_inactivity_interval_after_tackling(
            player,
            minNoOfInactCycles,
            maxNoOfInactCycles );

    int mini, maxi;

    get_player_inactivity_interval_after_being_fouled(
            player,
            mini,
            maxi );

    if( minNoOfInactCycles < mini )
        minNoOfInactCycles = mini;

    if( maxNoOfInactCycles < maxi )
        maxNoOfInactCycles = maxi;

    // store results in cache
    player_inactivities.minInact[    index ] = minNoOfInactCycles;
    player_inactivities.maxInact[    index ] = maxNoOfInactCycles;
    player_inactivities.validAtTime[ index ] = WSinfo::ws->time;
}

bool WSinfo::get_teammate( int number, PPlayer &player )
{
    player = get_teammate_by_number( number );

    if( player && player->alive ) // otherwise attentionto causes (wrong command form) messages
    {
        return true;
    }

    return false;
}

PPlayer WSinfo::get_teammate_by_number( int num )
{

    /** ATTENTION! The result can be 0, if no such valid palyer was found */
    /** ATTENTION! The result can be 0, if no such valid palyer was found */
    /** ATTENTION! The result can be 0, if no such valid palyer was found */

    if( num > NUM_PLAYERS || num < 1 )
    {
        //ERROR_OUT << " wrong teammate player number " << num;
        return 0;
    }

    return numbered_valid_players[ num ];
}

PPlayer WSinfo::get_opponent_by_number( int num )
{

    /** ATTENTION! The result can be 0, if no such valid palyer was found */
    /** ATTENTION! The result can be 0, if no such valid palyer was found */
    /** ATTENTION! The result can be 0, if no such valid palyer was found */

    if( num > NUM_PLAYERS || num < 1 )
    {
        if( num != 0 ) //just for the turnament in padova!
        //ERROR_OUT << " wrong opponent player number " << num;
        return 0;
    }

    return numbered_valid_players[num+NUM_PLAYERS];
}

int WSinfo::num_teammates_within_circle(const Vector &centre, const double radius)
{
    return (valid_teammates_without_me.keep_players_in_circle(centre, radius)).num;
}

double WSinfo::my_team_pos_of_offside_line()
{
    if(my_team_pos_of_offside_line_cache_ok) return my_team_pos_of_offside_line_cache;

    //consider players
    pset_tmp = WSinfo::valid_teammates;
    pset_tmp.keep_and_sort_players_by_x_from_left( 2 );

    //special consideration for my goalie
    PPlayer myGoalie = 0;

    if( WSinfo::ws->my_goalie_number != 0 ) myGoalie = WSinfo::get_teammate_by_number(WSinfo::ws->my_goalie_number);

    if( myGoalie && pset_tmp.get_player_by_number( WSinfo::ws->my_goalie_number ) == false )
    {
        //my goalie exists but is not within pset_tmp, so it must be outdated
        if( myGoalie->alive && pset_tmp.num > 0 && pset_tmp[ 0 ]->pos.getX() - myGoalie->pos.getX() > 5.0 )
        {
            pset_tmp.append( myGoalie );
        }
    }
    //end of special consideration of my goalie

    pset_tmp.keep_and_sort_players_by_x_from_left( 2 );
    if( pset_tmp.num == 0 )
    {
        my_team_pos_of_offside_line_cache = 0.0;
    }
    else
    {
        if( pset_tmp.num == 1 )
        {
            my_team_pos_of_offside_line_cache = 0.0;
        }
        else // pset_tmp.num == 2
        {
            my_team_pos_of_offside_line_cache = pset_tmp[ 1 ]->pos.getX();
        }
    }

    my_team_pos_of_offside_line_cache_ok = true;

    //take the ball into account
    if( WSinfo::is_ball_pos_valid() && WSinfo::ball->pos.getX() < my_team_pos_of_offside_line_cache )
    {
        my_team_pos_of_offside_line_cache = WSinfo::ball->pos.getX();
    }

    if( my_team_pos_of_offside_line_cache < -FIELD_BORDER_X )
    {
        my_team_pos_of_offside_line_cache = -FIELD_BORDER_X;
    }

    if( my_team_pos_of_offside_line_cache > 0.0 )
    {
        my_team_pos_of_offside_line_cache = 0.0;
    }

    return my_team_pos_of_offside_line_cache;
}

double WSinfo::his_team_pos_of_offside_line()
{
    if( his_team_pos_of_offside_line_cache_ok ) return his_team_pos_of_offside_line_cache;

    if( WSinfo::ws == NULL ) return 0.0;

    //consider players
    pset_tmp = WSinfo::valid_opponents;
    pset_tmp.keep_and_sort_players_by_x_from_right( 2 );

    //special consideration for my goalie
    PPlayer hisGoalie = 0;

    if( WSinfo::ws->his_goalie_number != 0 ) hisGoalie = WSinfo::alive_opponents.get_player_by_number(WSinfo::ws->his_goalie_number);
    if( hisGoalie && pset_tmp.get_player_by_number( WSinfo::ws->his_goalie_number ) == false )
    {
        //his goalie exists but is not within pset_tmp, so it must be outdated
        if( hisGoalie->alive && pset_tmp.num > 0 && hisGoalie->pos.getX() - pset_tmp[ 0 ]->pos.getX() > 5.0 )
        {
            pset_tmp.append( hisGoalie );
        }
    }
    //end of special consideration of my goalie
    
    pset_tmp.keep_and_sort_players_by_x_from_right( 2 );
    if( pset_tmp.num == 0 )
    {
        his_team_pos_of_offside_line_cache = 0.0;
    }
    else
    {
        if(pset_tmp.num == 1)
        {
            his_team_pos_of_offside_line_cache = 0.0;
        }
        else // pset_tmp.num == 2
        {
            his_team_pos_of_offside_line_cache = pset_tmp[ 1 ]->pos.getX();
        }
    }

    his_team_pos_of_offside_line_cache_ok = true;

    //take the ball into account
    if( WSinfo::is_ball_pos_valid() && WSinfo::ball->pos.getX() > his_team_pos_of_offside_line_cache )
    {
        his_team_pos_of_offside_line_cache = WSinfo::ball->pos.getX();
    }

    if( his_team_pos_of_offside_line_cache > FIELD_BORDER_X )
    {
        his_team_pos_of_offside_line_cache = FIELD_BORDER_X;
    }

    if( his_team_pos_of_offside_line_cache < 0.0 )
    {
        his_team_pos_of_offside_line_cache = 0.0;
    }

    return his_team_pos_of_offside_line_cache;
}

PPlayer WSinfo::get_teammate_with_newest_pass_info()
{
    if( !teammate_with_newest_pass_info_ok )
    {
        teammate_with_newest_pass_info = valid_teammates_without_me.get_player_with_newest_pass_info();
        teammate_with_newest_pass_info_ok = true;
    }
    return teammate_with_newest_pass_info;
}

PPlayer WSinfo::teammate_closest2ball()
{
    return valid_teammates.closest_player_to_point( ball->pos );
}

void WSinfo::visualize_state()
{
    if( LogOptions::is_off() ) return;

//    const double PLAYER_RADIUS = 1.8;
//    const double BALL_RADIUS   = 1.2;
    const double PLAYER_RADIUS = 1.4;
#if LOGGING && BASIC_LOGGING
    const double BALL_RADIUS   = .5;
    const int LOG_LEVEL       = 0;//2

//    const char   *MY_COLORS[] = { "0000ff", "000070", "000050", "000010" };
//    const char  *HIS_COLORS[] = { "ff0000", "700000", "500000", "100000" };
    const char   *MY_COLORS[] = { "0000ff", "0000f0", "000050", "000010" };
//    const char  *HIS_COLORS[] = { "ff0000", "f00000", "500000", "100000" };
    const char  *HIS_COLORS[] = { "ffff00", "f0f000", "505000", "100000" };
    const char *BALL_COLORS[] = { "ff66ff", "cc66cc", "996699", "666666", "336633" };

//    const int PCOL_INT[] = { 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0 };
    const int PCOL_INT[] = { 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
#endif

    if( ws->time == 0 ) return;

    int age;
    PPlayer player;

    PlayerSet pset = WSinfo::valid_teammates;
    pset += WSinfo::valid_opponents;

    for( int i = 0; i < pset.num; i++ )
    {
        player = pset[i]; //shortcut

        age = player->age;

        if( age > 13 )
        {
            age = 13;
        }

#if LOGGING && BASIC_LOGGING
        char const *color = "ff0000";

        if( player->team == MY_TEAM )
        {
            color = MY_COLORS[ PCOL_INT[ age ] ];
        }
        else
        {
            if( player->team == HIS_TEAM ){
                color = HIS_COLORS[ PCOL_INT[ age ] ];
            }
        }
#endif

        LOG_MDP( LOG_LEVEL, << _2D << VC2D( player->pos, PLAYER_RADIUS, color ) );
//        LOG_MDP(LOG_LEVEL,<< _2D << STRING2D( player->pos.x + PLAYER_RADIUS, player->pos.y, player->number << ", a=(" << player->age << ',' << player->age_vel << ',' << player->age_ang << ')', color ) );

        //orientation of player

        Vector tmp;
        tmp.init_polar( PLAYER_RADIUS * .75, player->ang );
        tmp += player->pos;

        LOG_MDP( LOG_LEVEL, << _2D << VL2D( player->pos, tmp, color ) );

/*
        if( player->vel.sqr_norm() > SQUARE( 0.2 ) )
        {
            LOG_MDP( LOG_LEVEL, << _2D << L2D( pos.x, pos.y, pos.x + 5 * player->vel.x, pos.y + 5 * player->vel.y, color ) );
            //LOG_MDP( LOG_LEVEL, << _2D << STRING2D( pos.x, pos.y + PLAYER_RADIUS, ( ( (int) (100.0 * player->vel.x ) ) / 100.0 ) << "," << ( (int) (100.0 * player->vel.y ) / 100.0 ) << "=" << player->vel.norm(), color ) );
        }
*/

        tmp = player->vel;
        tmp.normalize( PLAYER_RADIUS );
        tmp += player->pos;

        Vector tmp1 = player->vel;
        tmp1.normalize( 1.0 ); // max.velocity

        LOG_MDP( LOG_LEVEL, << _2D << VL2D( tmp, tmp + tmp1, "grey" ) );

        LOG_MDP( LOG_LEVEL, << _2D << VL2D( tmp, tmp + player->vel, "orange" ) );

    }

    if( is_ball_pos_valid() )
    {
        LOG_MDP( LOG_LEVEL, << _2D << VC2D( ws->ball.pos, BALL_RADIUS, BALL_COLORS[0] ) << VC2D( ws->ball.pos, BALL_RADIUS + .2, BALL_COLORS[ 0 ] ) );
        LOG_MDP( LOG_LEVEL, << _2D << VL2D( ws->ball.pos, ws->ball.pos + 10 * ws->ball.vel, BALL_COLORS[ 0 ] ) );
    }

#if LOGGING && BASIC_LOGGING
    double line = my_team_pos_of_offside_line();
#endif
    LOG_MDP( LOG_LEVEL, << _2D << L2D( line, -0.5 * ServerOptions::pitch_width, line, 0.5 * ServerOptions::pitch_width, "#006600" ) );

#if LOGGING && BASIC_LOGGING
    line = his_team_pos_of_offside_line();
#endif
    LOG_MDP( LOG_LEVEL, << _2D << L2D( line, -0.5 * ServerOptions::pitch_width, line, 0.5 * ServerOptions::pitch_width, "#006600" ) );
}

void WSinfo::set_relevant_teammates_default()
{
    num_relevant_teammates = 0; // reset

    switch( ClientOptions::player_no )
    {
    case 2:
        set_relevant_teammates( 6, 7, 3, 9 );
        break;
    case 3:
        set_relevant_teammates( 6, 7, 2, 4 );
        break;
    case 4:
        set_relevant_teammates( 8, 7, 5, 3 );
        break;
    case 5:
        set_relevant_teammates( 8, 7, 4, 11 );
        break;
    case 6:
        set_relevant_teammates( 9, 10, 7, 3, 2 );
        break;
    case 7:
        set_relevant_teammates( 10, 6, 8, 9, 11, 3, 4 );
        break;
    case 8:
        set_relevant_teammates( 11, 10, 7, 5, 4 );
        break;
    case 9:
        set_relevant_teammates( 10, 6, 7, 11, 8 );
        break;
    case 10:
        set_relevant_teammates( 11, 9, 7, 6, 8 );
        break;
    case 11:
        set_relevant_teammates( 10, 8, 7, 9, 6 );
        break;
    }
}

void WSinfo::set_relevant_teammates( const int t1, const int t2,
                                     const int t3, const int t4,
                                     const int t5, const int t6,
                                     const int t7, const int t8,
                                     const int t9, const int t10,
                                     const int t11)
{
    num_relevant_teammates = 0; // reset

    if(  t1 > 0 ) relevant_teammate[ num_relevant_teammates++ ] =  t1;
    if(  t2 > 0 ) relevant_teammate[ num_relevant_teammates++ ] =  t2;
    if(  t3 > 0 ) relevant_teammate[ num_relevant_teammates++ ] =  t3;
    if(  t4 > 0 ) relevant_teammate[ num_relevant_teammates++ ] =  t4;
    if(  t5 > 0 ) relevant_teammate[ num_relevant_teammates++ ] =  t5;
    if(  t6 > 0 ) relevant_teammate[ num_relevant_teammates++ ] =  t6;
    if(  t7 > 0 ) relevant_teammate[ num_relevant_teammates++ ] =  t7;
    if(  t8 > 0 ) relevant_teammate[ num_relevant_teammates++ ] =  t8;
    if(  t9 > 0 ) relevant_teammate[ num_relevant_teammates++ ] =  t9;
    if( t10 > 0 ) relevant_teammate[ num_relevant_teammates++ ] = t10;
    if( t11 > 0 ) relevant_teammate[ num_relevant_teammates++ ] = t11;
}


void WSinfo::set_current_opponent_identifier( char id )
{
    cvCurrentOpponentIdentifier = id;
}
char WSinfo::get_current_opponent_identifier()
{
    return cvCurrentOpponentIdentifier;
}
