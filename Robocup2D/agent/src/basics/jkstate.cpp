#include "jkstate.h"

#define LOG_LEVEL 0
#define DBLOG_POL(LLL,XXX) LOG_DEB(LLL,<<"Score04: "<<XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_DEB(LLL,<<_2D<<XXX)

Score04Action::Score04Action()
{

}

void Score04Action::set_pass( int closeness )
{
    if( closeness < 1 )
        closeness = 1;
    if( closeness > 2 )
        closeness = 2;
    type = closeness;
    param = 0.0;
}

void Score04Action::set_pass( int closeness, Vector where, int player )
{
    if( closeness < 1 )
        closeness = 1;
    if( closeness > 2 )
        closeness = 2;
    type = closeness;
    param = 1.0;
    param3 = player;
    target = where;
}

void Score04Action::set_dribble()
{
    type = 3;
    param = .0;
}

void Score04Action::set_immediate_selfpass( float dir )
{
    type = 4;
    param = dir;
}

void Score04Action::set_selfpass( float dir, float speed, Vector target_pos )
{
    type = 5;
    param = dir;
    param2 = speed;
    target = target_pos;
}

Score04State::Score04State()
{
}

bool Score04State::fromString( string input )
{
    istringstream iss( input );
    iss >> ball_dist;
    iss >> ball_ang;
    iss >> ball_vel_norm;
    iss >> ball_vel_ang;
    iss >> goal_dist;
    iss >> goal_ang;
    for( int i = 0; i < NUM_OPP; i++ )
    {
        iss >> opp_dist[ i ];
        iss >> opp_ang[ i ];
    }
    for( int i = 0; i < NUM_FRIENDS; i++ )
    {
        iss >> friend_dist[ i ];
        iss >> friend_ang[ i ];
    }
    return true;
}

string Score04State::toString()
{
    ostringstream os;
    os << ball_dist << " " << ball_ang << " " << ball_vel_norm << " "
            << ball_vel_ang << " " << goal_dist << " " << goal_ang << " ";
    for( int i = 0; i < NUM_OPP; i++ )
        os << opp_dist[ i ] << " " << opp_ang[ i ] << " ";
    for( int i = 0; i < NUM_FRIENDS; i++ )
        os << friend_dist[ i ] << " " << friend_ang[ i ] << " ";
    os << ends;
    return os.str();
}

jkState::jkState()
{

}

Vector jkState::get_opp_pos( int i )
{
    if( ! ( i < NUM_OPP ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    return opp_pos[ i ];
}

Vector jkState::get_opp_vel( int i )
{
    if( ! ( i < NUM_OPP ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    return opp_vel[ i ];
}

ANGLE jkState::get_opp_ang( int i )
{
    if( ! ( i < NUM_OPP ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    return opp_ang[ i ];
}

Vector jkState::get_friend_pos( int i )
{
    if( ! ( i < NUM_FRIENDS ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    return friend_pos[ i ];
}

Vector jkState::get_friend_vel( int i )
{
    if( ! ( i < NUM_FRIENDS ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    return friend_vel[ i ];
}

ANGLE jkState::get_friend_ang( int i )
{
    if( ! ( i < NUM_FRIENDS ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    return friend_ang[ i ];
}

Vector jkState::get_ball_pos()
{
    return ball_pos;
}
Vector jkState::get_ball_vel()
{
    return ball_vel;
}
Vector jkState::get_my_pos()
{
    return my_pos;
}
Vector jkState::get_my_vel()
{
    return my_vel;
}
ANGLE jkState::get_my_ang()
{
    return my_ang;
}

void jkState::set_opp_pos( int i, Vector what )
{
    if( ! ( i < NUM_OPP ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    opp_pos[ i ] = what;
}

void jkState::set_opp_vel( int i, Vector what )
{
    if( ! ( i < NUM_OPP ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    opp_vel[ i ] = what;
}

void jkState::set_opp_ang( int i, ANGLE what )
{
    if( ! ( i < NUM_OPP ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    opp_ang[ i ] = what;
}

void jkState::set_friend_pos( int i, Vector what )
{
    if( ! ( i < NUM_FRIENDS ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    friend_pos[ i ] = what;
}

void jkState::set_friend_vel( int i, Vector what )
{
    if( ! ( i < NUM_FRIENDS ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    friend_vel[ i ] = what;
}

void jkState::set_friend_ang( int i, ANGLE what )
{
    if( ! ( i < NUM_FRIENDS ) )
    {
        cerr << "jkState: ARRAY OUT OF BOUNDS ERROR\n";
        DBLOG_POL( LOG_LEVEL, "jkState: ARRAY OUT OF BOUNDS ERROR\n" );
    }
    friend_ang[ i ] = what;
}

void jkState::set_ball_pos( Vector what )
{
    ball_pos = what;
}
void jkState::set_ball_vel( Vector what )
{
    ball_vel = what;
}
void jkState::set_my_pos( Vector what )
{
    my_pos = what;
}
void jkState::set_my_vel( Vector what )
{
    my_vel = what;
}
void jkState::set_my_ang( ANGLE what )
{
    my_ang = what;
}

void jkState::resort_opps()
{ //sort opponents acc. to their distance to "my_pos", useful e.g. after ballholder changed
    Vector opp_pos2[ NUM_OPP ];  //temporary
    Vector opp_vel2[ NUM_OPP ];
    ANGLE opp_ang2[ NUM_OPP ];
    Sort * eval_pos = new Sort( NUM_OPP );
    for( int i = 0; i < NUM_OPP; i++ )
    {
        opp_pos2[ i ] = opp_pos[ i ];
        opp_vel2[ i ] = opp_vel[ i ];
        opp_ang2[ i ] = opp_ang[ i ];
        eval_pos->add( i, opp_pos[ i ].distance( my_pos ) );
    }
    eval_pos->do_sort();  //smallest distance first
    int ind;
    for( int i = 0; i < NUM_OPP; i++ )
    {
        ind = eval_pos->get_key( i );
        opp_pos[ i ] = opp_pos2[ ind ];
        opp_vel[ i ] = opp_vel2[ ind ];
        opp_ang[ i ] = opp_ang2[ ind ];
    }
}

void jkState::resort_friends()
{  //sort friends acc. to their distance to "my_pos"
    Vector friend_pos2[ NUM_FRIENDS ];  //temporary
    Vector friend_vel2[ NUM_FRIENDS ];
    ANGLE friend_ang2[ NUM_FRIENDS ];
    Sort * eval_pos = new Sort( NUM_FRIENDS );
    for( int i = 0; i < NUM_FRIENDS; i++ )
    {
        friend_pos2[ i ] = friend_pos[ i ];
        friend_vel2[ i ] = friend_vel[ i ];
        friend_ang2[ i ] = friend_ang[ i ];
        eval_pos->add( i, friend_pos[ i ].distance( my_pos ) );
    }
    eval_pos->do_sort();  //smallest distance first
    int ind;
    for( int i = 0; i < NUM_FRIENDS; i++ )
    {
        ind = eval_pos->get_key( i );
        friend_pos[ i ] = friend_pos2[ ind ];
        friend_vel[ i ] = friend_vel2[ ind ];
        friend_ang[ i ] = friend_ang2[ ind ];
    }
}

void jkState::debug_out( char* color )
{
    DBLOG_DRAW( LOG_LEVEL, VC2D(my_pos,0.5, color) );
    DBLOG_DRAW( LOG_LEVEL, VC2D(ball_pos,0.7, color) );
    for( int i = 0; i < NUM_OPP; i++ )
        DBLOG_DRAW( LOG_LEVEL, VC2D(opp_pos[i],0.3, color) );
    for( int i = 0; i < NUM_FRIENDS; i++ )
        DBLOG_DRAW( LOG_LEVEL, VC2D(friend_pos[i],0.1, color) );
}

MyState jkState::get_old_version_State()
{ //from the times where a state consisted only of me and one opp.
    MyState tmp;
    tmp.my_pos = my_pos;
    tmp.my_vel = my_vel;
    tmp.my_angle = my_ang;
    tmp.ball_pos = ball_pos;
    tmp.ball_vel = ball_vel;
    tmp.op_pos = opp_pos[ 1 ]; //1 = consider closest player (0 goalie)
//   tmp.op_vel=opp_vel[1];
    tmp.op_bodydir = opp_ang[ 1 ];
    tmp.op = NULL;                //not used
    tmp.op_bodydir_age = 0; //not used
    return tmp;
}

void jkState::copy_to( jkState &target_state )
{
    target_state.my_pos = my_pos;
    target_state.my_vel = my_vel;
    target_state.my_ang = my_ang;
    target_state.ball_pos = ball_pos;
    target_state.ball_vel = ball_vel;
    for( int i = 0; i < NUM_OPP; i++ )
    {
        target_state.opp_pos[ i ] = opp_pos[ i ];
        target_state.opp_vel[ i ] = opp_vel[ i ];
        target_state.opp_ang[ i ] = opp_ang[ i ];
    }
    for( int i = 0; i < NUM_FRIENDS; i++ )
    {
        target_state.friend_pos[ i ] = friend_pos[ i ];
        target_state.friend_vel[ i ] = friend_vel[ i ];
        target_state.friend_ang[ i ] = friend_ang[ i ];
    }
}

float jkState::angle_player2player( Vector relative_to, Vector oppt,
        float rel_to_norm )
{
    Vector p2p = oppt - relative_to; //Cosinus zwichen 2 Vektoren ist  Skalarprodukt(a,b) / |a|*|b|
    double result = ( relative_to.getX() * p2p.getX() + relative_to.getY() * p2p.getY() ) / ( rel_to_norm * p2p.norm() );
    if( ! ( result > -10000000.0 ) ) //should be !false only with NAN
        return 0.0;  //use 0.0 instead of NAN
    return result;  //return Cosinus \in [-1,1]
    //acosf((relative_to.x*p2p.x+relative_to.y*p2p.y)/(rel_to_norm*p2p.norm()));
}

float jkState::scale_dist( float input_dist )
{
    return ( Tools::min( input_dist, 20.0 ) - 10.0 ) / 10.0; //maximum distance of 20, scale to [-1,1]
}

Score04State jkState::get_scoreState()
{  //change representation to a format usable by a neural net
    Score04State cst;
    int i;
    float my_norm = my_pos.norm();  //just to save some calculation time
    for( i = 1; i < NUM_OPP; i++ )
    {
        cst.opp_dist[ i ] = scale_dist( my_pos.distance( opp_pos[ i ] ) );
        cst.opp_ang[ i ] = angle_player2player( my_pos, opp_pos[ i ], my_norm );
    }
    for( i = 0; i < NUM_FRIENDS; i++ )
    {
        cst.friend_dist[ i ] = scale_dist( my_pos.distance( friend_pos[ i ] ) );
        cst.friend_ang[ i ] = angle_player2player( my_pos, friend_pos[ i ], my_norm );
    }
    cst.goal_dist = scale_dist( my_pos.distance( Vector( 52.5, 0.0 ) ) );
    cst.goal_ang = angle_player2player( my_pos, Vector( 52.5, 0.0 ), my_norm );
    cst.opp_dist[ OPP_GOALIE ] = scale_dist( my_pos.distance( opp_pos[ 0 ] ) );
    cst.opp_ang[ OPP_GOALIE ] = angle_player2player( my_pos, opp_pos[ 0 ], my_norm );
    cst.ball_dist = scale_dist( my_pos.distance( ball_pos ) );
    cst.ball_ang = angle_player2player( my_pos, ball_pos, my_norm );
    cst.ball_vel_norm = ( ball_vel.norm() - ServerOptions::ball_speed_max / 2.0 ) / ServerOptions::ball_speed_max / 2.0; //max ballspeed is 2.7, scale to [-1,1]
    cst.ball_vel_ang = angle_player2player( my_pos, ball_vel, my_norm );
    return cst;
}

bool jkState::fromString( string input )
{
    double tmp;
    istringstream iss( input );
    iss >> tmp;
    my_pos.setX( tmp );
    iss >> tmp;
    my_pos.setY( tmp );
    iss >> tmp;
    my_vel.setX( tmp );
    iss >> tmp;
    my_vel.setY( tmp );
    iss >> tmp;
    my_ang.set_value( tmp );
    iss >> tmp;
    ball_pos.setX( tmp );
    iss >> tmp;
    ball_pos.setY( tmp );
    iss >> tmp;
    ball_vel.setX( tmp );
    iss >> tmp;
    ball_vel.setY( tmp );
    for( int i = 0; i < NUM_OPP; i++ )
    {
        iss >> tmp;
        opp_pos[ i ].setX( tmp );
        iss >> tmp;
        opp_pos[ i ].setY( tmp );
        iss >> tmp;
        opp_vel[ i ].setX( tmp );
        iss >> tmp;
        opp_vel[ i ].setY( tmp );
        iss >> tmp;
        opp_ang[ i ].set_value( tmp );
    }
    for( int i = 0; i < NUM_FRIENDS; i++ )
    {
        iss >> tmp;
        friend_pos[ i ].setX( tmp );
        iss >> tmp;
        friend_pos[ i ].setY( tmp );
        iss >> tmp;
        friend_vel[ i ].setX( tmp );
        iss >> tmp;
        friend_vel[ i ].setY( tmp );
        iss >> tmp;
        friend_ang[ i ].set_value( tmp );
    }
    return true;
}

string jkState::toString()
{
    ostringstream os;
    os << my_pos.getX() << " " << my_pos.getY() << " " << my_vel.getX() << " "
            << my_vel.getY() << " " << my_ang << " " << ball_pos.getX() << " "
            << ball_pos.getY() << " " << ball_vel.getX() << " "
            << ball_vel.getY() << " ";
    for( int i = 0; i < NUM_OPP; i++ )
        os << opp_pos[ i ].getX() << " " << opp_pos[ i ].getY() << " "
                << opp_vel[ i ].getX() << " " << opp_vel[ i ].getY() << " "
                << opp_ang[ i ] << " ";
    for( int i = 0; i < NUM_FRIENDS; i++ )
        os << friend_pos[ i ].getX() << " " << friend_pos[ i ].getY() << " "
                << friend_vel[ i ].getX() << " " << friend_vel[ i ].getY()
                << " " << friend_ang[ i ] << " ";
    os << ends;
    return os.str();
}

void jkState::get_from_WS()
{
    get_from_WS( WSinfo::me );
}

void jkState::get_from_WS( PPlayer me )
{
    get_from_WS( me, me->pos );
}

int jkState::rand_in_range( int a, int b )
{
    int off = ( int ) round( drand48() * 32767 );
    return a + ( off % ( b - a + 1 ) );
}

void jkState::get_from_WS( PPlayer me, Vector next_pos ) //might be used to simulate viewpoint of another player
{
    my_pos = next_pos;                        //("sich in jemand reinversetzen")
    my_vel = me->vel;
    my_ang = me->ang;

    int i;
    PlayerSet opp = WSinfo::alive_opponents;
    if( WSinfo::his_goalie != NULL )
        opp.remove( WSinfo::his_goalie );
    opp.keep_and_sort_closest_players_to_point( NUM_OPP - 1, me->pos );
    for( i = 1; i < NUM_OPP; i++ )
    {  //array space 0 reserved for goalie
        if( opp.num < i - 1 )
        {
            opp_pos[ 0 ] = Vector( -52.0, 0.0 );
            opp_vel[ 0 ] = Vector( .0, .0 );
            opp_ang[ 0 ] = ANGLE( 0 );
        }
        else
        {
            opp_pos[ i ] = opp[ i - 1 ]->pos;
            opp_vel[ i ] = opp[ i - 1 ]->vel;
            opp_ang[ i ] = opp[ i - 1 ]->ang;
        }
    }
    PlayerSet team = WSinfo::alive_teammates;
    team.remove( me );
    team.keep_and_sort_closest_players_to_point( team.num, me->pos );
    int j = 0;
    int count = 0;
    do
    {                                        //determine number of close players
        if( team[ j ]->pos.sqr_distance( me->pos ) < 25 )
            count++;
    } while( j++ < team.num );
    while( count > NUM_FRIENDS )
    {                   //if too many close players remove some players randomly
        team.remove( team[ rand_in_range( 0, count - 1 ) ] );
        count--;
    }
    for( i = 0; i < NUM_FRIENDS; i++ )
    {
        friend_pos[ i ] = team[ i ]->pos;
        friend_vel[ i ] = team[ i ]->vel;
        friend_ang[ i ] = team[ i ]->ang;
    }
    if( WSinfo::his_goalie != NULL )
    {
        opp_pos[ 0 ] = WSinfo::his_goalie->pos;
        opp_vel[ 0 ] = WSinfo::his_goalie->vel;
        opp_ang[ 0 ] = WSinfo::his_goalie->ang;
    }
    else
    {
        opp_pos[ 0 ] = Vector( -52.0, 0.0 );
        opp_vel[ 0 ] = Vector( .0, .0 );
        opp_ang[ 0 ] = ANGLE( 0 );
    }
    ball_pos = WSinfo::ball->pos;
    ball_vel = WSinfo::ball->vel;
}
