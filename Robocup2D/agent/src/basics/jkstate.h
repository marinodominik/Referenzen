#ifndef _JKSTATE_H_
#define _JKSTATE_H_

#include <stdlib.h>
#include "Vector.h"
#include "ws_info.h"
#include "tools.h"
#include "log_macros.h"

#include <sstream>

#include "Cmd.h"
#include "valueparser.h"
#include "sort.h"

#define NUM_OPP 5      //number of opponents (e.g. 5 : 4 + goalie)
#define OPP_GOALIE 0 //array number of goalie
#define NUM_FRIENDS 2 //is not really changeable since some stuff hard coded, sorry

class Score04Action
{
public:
    int type;          //type of action:
    /*
     1 : pass_to_closest_friend
     2 : pass_to_snd_closest_friend
     3 : dribblestraight
     4 : immediate_selfpass
     (takes a parameter for the direction)
     5 : selfpass
     (takes parameter for direction, ball_speed, target_pos)
     */
    float param;   //some actions might need a parameter
    float param2; //two
    int param3;    //even three
    Vector target; //or a target

    Score04Action();

    void set_pass( int closeness );
    void set_pass( int closeness, Vector where, int player );
    void set_dribble();
    void set_immediate_selfpass( float dir );
    void set_selfpass( float dir, float speed, Vector target_pos );
};

class Score04State //used as input for the net
{
public:
    float opp_dist[ NUM_OPP ];  //polar coordinates of opps
    float opp_ang[ NUM_OPP ];
    float friend_dist[ NUM_FRIENDS ];  //polar coordinates of friends
    float friend_ang[ NUM_FRIENDS ];
    float goal_dist;  //polar coordinates of goal (middle)
    float goal_ang;
    float ball_dist;  //polar coordinates of ball
    float ball_ang;
    float ball_vel_norm;  //velocity of ball
    float ball_vel_ang;

    Score04State();

    bool fromString( string input );
    string toString();
};

class jkState
{
public:

    Vector opp_pos[ NUM_OPP ];  //coordinates of opps
    Vector opp_vel[ NUM_OPP ];
    ANGLE opp_ang[ NUM_OPP ];
    Vector friend_pos[ NUM_FRIENDS ];  //coordinates of friends
    Vector friend_vel[ NUM_FRIENDS ];
    ANGLE friend_ang[ NUM_FRIENDS ];
    Vector ball_pos;  //coordinates of ball
    Vector ball_vel;
    Vector my_pos;
    Vector my_vel;
    ANGLE my_ang;

    jkState();

    Vector get_opp_pos( int i );
    Vector get_opp_vel( int i );
    ANGLE get_opp_ang( int i );
    Vector get_friend_pos( int i );
    Vector get_friend_vel( int i );
    ANGLE get_friend_ang( int i );
    Vector get_ball_pos();
    Vector get_ball_vel();
    Vector get_my_pos();
    Vector get_my_vel();
    ANGLE get_my_ang();

    void set_opp_pos( int i, Vector what );
    void set_opp_vel( int i, Vector what );
    void set_opp_ang( int i, ANGLE what );
    void set_friend_pos( int i, Vector what );
    void set_friend_vel( int i, Vector what );
    void set_friend_ang( int i, ANGLE what );
    void set_ball_pos( Vector what );
    void set_ball_vel( Vector what );
    void set_my_pos( Vector what );
    void set_my_vel( Vector what );
    void set_my_ang( ANGLE what );
    void resort_opps(); //sort opponents acc. to their distance to "my_pos", useful e.g. after ballholder changed
    void resort_friends(); //sort friends acc. to their distance to "my_pos"
    void debug_out( char* color );

    MyState get_old_version_State(); //from the times where a state consisted only of me and one opp.

    void copy_to( jkState &target_state );

    float angle_player2player( Vector relative_to, Vector oppt,
            float rel_to_norm );

    float scale_dist( float input_dist );

    Score04State get_scoreState(); //change representation to a format usable by a neural net

    bool fromString( string input );
    string toString();

    void get_from_WS();
    void get_from_WS( PPlayer me );
    void get_from_WS( PPlayer me, Vector next_pos ); //might be used to simulate viewpoint of another player ("sich in jemand reinversetzen")

    int rand_in_range( int a, int b );
};

#endif
