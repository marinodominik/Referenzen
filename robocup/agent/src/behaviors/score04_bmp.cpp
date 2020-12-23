/* Author: Johannes Knabe, 2004
 *
 * This is a behaviour for scoring, i.e. behaving in the last meters to the goal
 * in a way that maximises our goal chances.
 *
 * If you are interested in how the learning works: The agents are only writing
 * their data to tmp... files which are then build together by the coach
 * -> have a look at the coach module: mod_train_jk.{h,c}
 *
 */
#include "score04_bmp.h"
//#include "../policy/planning.h"
//#include "../policy/policy_tools.h"
//#include "../policy/positioning.h"
//#include "Vector.h"
//#include "blackboard.h"
//#include "log_macros.h"
//#include "mdp_info.h"
//#include "options.h"
#include "score04_tools.h"
//#include "tools.h"
//#include "valueparser.h"
//#include "ws.h"
//#include "ws_info.h"
//#include "ws_memory.h"
//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>

//#include "../basics/PlayerSet.h"
//#include "geometry2d.h";

//read parameters from net
//#define DELTA_NULL .4      //learnrate
//#define DELTA_MAX 1.2     //momentum
//#define WEIGHT_DECAY 0 //weight decay

//Note that never absolute values should be used - this value makes sense only in a
//relative fashion, i.e. average
#define REJECT_VAL 0.999707 //if initial states evaluation is higher than that: refuse playing,
//set to a value higher 1.0 to disable and allways accept

//read from net nowadays
//4 layers are used, where the last has only one neuron for output
//the definitions in here are normally not used since nets are loaded with their architecture
//#define NUM_FEATS 20 //this should equal 6+NUM_OPP*2+NUM_FRIENDS*2 (see jkstate.h)
//#define NUM_HIDDEN1 20
//#define NUM_HIDDEN2 20

#define MAX_PREDICTION_DEPTH 3 //higher is better but may take too much time

#define CMD_NOT_SET 999
#define LOOSE_SITUATION 9999

#define LOG_LEVEL 0
#define DEBUG 1 //lots of logging - turn of for fast play
#define DBLOG_POL(LLL, XXX) LOG_DEB (LLL, << "Score04: " << XXX)
#define DBLOG_DRAW(LLL, XXX) LOG_DEB (LLL, << _2D << XXX)

float exploration = 0.2;     //by default exploration rate of 20 percent
int exploration_num = -1; //by default do exploration with above rate (if >0 only exploration iff exploration_num==WSinfo::me->number)
float immediate_cost = 0.01; //would allow for a maximum of 100 steps until costs equal 1
bool use_me;
bool train_mode;
int number_of_attackers;
int number_of_defenders;
int random_mode;
bool model_mode;
bool use_me_nonballholder;
bool use_model_ballholder;
bool use_model_nonballholder;
bool use_exploration_nonballholder;
bool use_V_for_actions;
bool cost_per_action;
char load_net[ 500 ];
char func_map_file[ 500 ];
char save_nets[ 500 ];
char save_prots[ 500 ];
Prot prot_memory[ MAX_PROT_MEM ];
int prot_mem_count = 0;
int ball_holder;
int holding_ball_for = 0;
int last_teammate_excl_me_at_ball = -1;
Vector cached_no_ball_pos = Vector( .0, -1. );

Sort *eval_actions;
int eval_action_cnt = 0;
bool Score04::initialized = false;
Net *Score04::net;

Score04::Score04()
{
    score04TriedCounter = 0;
    score04ExecutedCounter = 0;

    go2pos = new NeuroGo2Pos();
    basiccmd = new BasicCmd();
    onetwokick = new OneOrTwoStepKick();
    neurokick = new NeuroKick05();
    intercept = new NeuroIntercept();
    jkwball03 = new JKWball03();
    dribblestraight = new DribbleStraight();
    selfpass = new Selfpass();
    last_called = -1;
    last_good_pos = Vector( 60.0, 60.0 );
    sequence_number = 0;
    //reset_intention(false);
}

Score04::~Score04()
{
    delete go2pos;
    delete basiccmd;
    delete onetwokick;
    delete selfpass;
    delete intercept;
    delete jkwball03;
    delete dribblestraight;
}

//returns random integer in range [a,b] \in Z
int rand_in_range( int a, int b )
{
    int off = round( drand48() * 32767 );
    return a + ( off % ( b - a + 1 ) );
}

double max( double a, double b )
{
    if( a > b )
        return a;
    return b;
}

double min( double a, double b )
{
    if( a < b )
        return a;
    return b;
}

void Score04::memorize( int action )
{
    if( prot_mem_count == MAX_PROT_MEM )
        return;
    prot_memory[ prot_mem_count ].time = WSinfo::ws->time;
    jkState tmp;
    tmp.get_from_WS();
    prot_memory[ prot_mem_count ].stat = tmp.get_scoreState();
    prot_memory[ prot_mem_count++ ].action = action;
}

int Score04::exit_num()
{
    //jk hack to get only a certain number of attackers,
    //all players with a number smaller than the returned
    //will be killed in the very beginning
    if( ( number_of_attackers == -1 ) || ( !use_me ) )
        return -1; //cases in which we do nothing
    if( ( !train_mode ) && ( exploration > 0.0 ) )
        return -1; //not training and not statistics: all play!
    return 11 - number_of_attackers;
}

bool Score04::protocol()
{
    if( !train_mode )
        return false;
    FILE *protfile;
    char pfname[ 200 ];
    strcpy( pfname, save_prots );
    strcat( pfname, "tmp_prot_" );
    strcat( pfname, inttostr( WSinfo::me->number ).c_str() );
    strcat( pfname, ".txt" );
    protfile = fopen( pfname, "w" ); //lateron construct more sophisticated filenames..
    if( protfile == NULL )
    {
        cerr << "\nProtocol file could not be opened!\n";
        return false;
    }
    for( int i = 0; i < prot_mem_count; i++ )
    {
        //cerr << prot_memory[i].toString() <<"\n";
        fputs( prot_memory[ i ].toString().c_str(), protfile );
        fputs( "\n", protfile );
    }
    prot_mem_count = 0;
    fclose( protfile );
    return true;
}

bool Score04::init( char const *conf_file, int argc, char const * const *argv )
{
    if( initialized )
        return true;
    //JK PASS_MSG_HACK begin
    if( WSinfo::me_full != NULL )
    {
        WSinfo::jk_pass_msg_set = false;
        strcpy( WSinfo::jk_pass_msg, "empty" );
    }
    //JK PASS_MSG_HACK end
    bool res = NeuroGo2Pos::init( conf_file, argc, argv ) && BasicCmd::init( conf_file, argc, argv ) && Selfpass::init( conf_file, argc, argv ) && NeuroIntercept::init( conf_file, argc, argv ) && DribbleStraight::init( conf_file, argc, argv ) && OneOrTwoStepKick::init( conf_file, argc, argv ) && JKWball03::init( conf_file, argc, argv ) && NeuroKick05::init( conf_file, argc, argv );

    net = new Net();
    train_mode = false;
    model_mode = true;
    use_me = false;
    use_me_nonballholder = false;
    use_model_ballholder = true;
    use_model_nonballholder = false;
    use_exploration_nonballholder = false;
    use_V_for_actions = true;
    cost_per_action = true;
    number_of_attackers = -1; //disable killing of nun-used players by default
    number_of_defenders = 1; //consider only one by default, not very good normally, see .conf file
    random_mode = 0; //by default uniform distribution, i.e. every action has same chance to get selected when randomizing,
                     //opposed to weighted distribution, where actions with higher Value have a proportionally higher chance of getting selected
    ValueParser vp( CommandLineOptions::policy_conf, "Score04_bmp" );
    vp.get( "use_score04", use_me );
    vp.get( "use_V_for_actions", use_V_for_actions );
    vp.get( "cost_per_action", cost_per_action );
    vp.get( "use_score04_for_nonballholders", use_me_nonballholder );
    vp.get( "good_model_nonballholder_succstate", use_model_nonballholder );
    vp.get( "use_exploration_nonballholder", use_exploration_nonballholder );
    vp.get( "exploration_num", exploration_num );
    vp.get( "model_ballholder_succstate", use_model_ballholder );
    vp.get( "train_mode", train_mode );
    vp.get( "exploration", exploration );
    vp.get( "model_mode", model_mode );

    vp.get( "number_of_attackers", number_of_attackers );
    vp.get( "number_of_defenders", number_of_defenders );

    vp.get( "random_mode", random_mode );
    if( train_mode )
    {
        vp.get( "save_nets", save_nets, 500 );
        vp.get( "save_prots", save_prots, 500 );
    }
    vp.get( "load_net", load_net, 500 );
    func_map_file[ 0 ] = '-'; //no snapshot of scattered function data by default
    vp.get( "func_map", func_map_file, 500 );
    if( load_net[ 0 ] != '-' )
    {
        if( net->load_net( load_net ) == FILE_ERROR )
        {
            cerr << "Score04: No net-file found in " << load_net << "\n";
            initialized = false;
            return false;
        }
        else
        { //nothing to do here, all info loaded from file
        }
    }
    else
    { //if load_net == "-" then start from scratch
        cerr << "Should never come here!!! (fatal err 3)\n";
        /*int layer_nodes[4];

         //OLD code for creating net

         learnparams[0] = DELTA_NULL;
         learnparams[1] = DELTA_MAX;
         learnparams[2] = WEIGHT_DECAY;

         layer_nodes[0]=NUM_FEATS;
         layer_nodes[1]=NUM_HIDDEN1;
         layer_nodes[2]=NUM_HIDDEN2;
         layer_nodes[3]=1;
         net->create_layers(4,layer_nodes);
         net->connect_layers();
         net->init_weights(0,.5); //set all connection weights to random values in [ -0.5,0.5]
         net->set_update_f(1,learnparams); //1 = RProp

         char temp[500];
         strcpy(temp,save_nets);
         strcat(temp,"test");

         strcat(temp,inttostr(1234));
         strcat(temp,".net");
         net->save_net(temp);*/
    }
    if( train_mode )
        std::cout << "\nScore04 in training mode.\n" << "Will start from net: "
                << load_net << "\nWill save generated nets to dir: "
                << save_nets << "\n";
    /* // little net test:
     if(train_mode){
     //float *result;
     float inp[NUM_FEATS];
     for(int i=0;i<NUM_FEATS;i++)
     inp[i]=1.0;
     net->forward_pass(inp,net->out_vec);
     std::cout<<"net test result: "<<net->out_vec[0];
     }
     */
    ball_holder = -1;
    initialized = true;
    return res;
}

float Score04::get_V_for_State( jkState state )
{
    //return 0.5;
    Score04State tmp = state.get_scoreState();
#ifdef DEBUG
    DBLOG_POL( LOG_LEVEL, "\nScore04State\n" << tmp.toString () << "\n" );
#endif
    float inp[ 6 + number_of_defenders * 2 + number_of_attackers + 2 ];
    inp[ 0 ] = tmp.ball_dist;
    inp[ 1 ] = tmp.ball_ang;
    inp[ 2 ] = tmp.ball_vel_norm;
    inp[ 3 ] = tmp.ball_vel_ang;
    inp[ 4 ] = tmp.goal_dist;
    inp[ 5 ] = tmp.goal_ang;
    for( int i = 0; i < number_of_defenders; i++ )
    {
        inp[ 6 + ( 2 * i ) ] = tmp.opp_dist[ i ];
        inp[ 6 + ( 2 * i ) + 1 ] = tmp.opp_ang[ i ];
    }
    for( int i = 0; i < number_of_attackers; i++ )
    {
        inp[ 6 + ( 2 * number_of_defenders ) + ( 2 * i ) ] = tmp.friend_dist[ i ];
        inp[ 6 + ( 2 * number_of_defenders ) + ( 2 * i ) + 1 ] = tmp.friend_ang[ i ];
    }
    net->forward_pass( inp, net->out_vec );
    //begin for debug only
    if( ! ( net->out_vec[ 0 ] > -10000000.0 ) )
    { //should be !false only with NAN
        Score04State tmp = state.get_scoreState();
        cerr << "\nNAN-ERROR\nGET_V_FOR_STATE says " << net->out_vec[ 0 ]
                << "\nto State: " << state.toString() << "\nas ScoreState:   "
                << tmp.toString() << "\n\n";
    }
    //end for debug only
    return net->out_vec[ 0 ];
}

void Score04::reset_intention( bool count )
{
    last_action_type = 0;
    last_action_time = -99;
    last_action_pass = false;
    last_action_pass_target_num = -1;
    last_action_pass_time = -99;
    holding_ball_for = 0;
    last_teammate_excl_me_at_ball = -1;
    //jkwball03->reset_intention();
    last_called = -2;
    ball_holder = -1;
}

bool Score04::get_cmd( Cmd &cmd )
{
#ifdef DEBUG
    DBLOG_POL( LOG_LEVEL, "- starts" );
#endif
    /*DISABLE FOR ANY SERIOUS PLAYING!!!
     //begin calculate average output of net function
     if(WSinfo::me->number==10){
     jkState state;
     state.fromString("30.0 0.0 0.0 0.0 0.0    30.5 0.0 0.0 0.0    50.0 0.0 0.0 0.0 0.0    40.0 -10.0 0.0 0.0 0.0    40.0 10.0 0.0 0.0 0.0    40.0 20.0 0.0 0.0 0.0    40.0 -20.0 0.0 0.0 0.0    30.0 -10.0 0.0 0.0 0.0    30.0 10.0 0.0 0.0 0.0");
     double sum=.0;
     int num=0;
     for(int i=0;i<100000;i++)
     { //generate random state; would be much better to have realistic data here!!!
     for(int j=0;j<NUM_OPP;j++){
     state.opp_pos[j].x=44+(8*(drand48()-0.5)); // [40..48]
     state.opp_pos[j].y=(14*(drand48()-0.5)); // [-7..7]
     state.opp_vel[j].x=drand48();
     state.opp_vel[j].y=drand48();
     }
     for(int j=0;j<NUM_FRIENDS;j++){
     state.friend_pos[j].x=38+(8*(drand48()-0.5)); // [34..42]
     state.friend_pos[j].y=(30*(drand48()-0.5)); // [-15..15]
     state.friend_vel[j].x=drand48();
     state.friend_vel[j].y=drand48();
     }
     state.ball_pos.x=38+(8*(drand48()-0.5)); // [34..42]
     state.ball_pos.y=(30*(drand48()-0.5)); // [-15..15]
     state.ball_vel.x=drand48();
     state.ball_vel.y=drand48();
     state.my_pos.x=state.ball_pos.x;
     state.my_pos.y=state.ball_pos.y;
     state.my_vel.x=state.ball_vel.x;
     state.my_vel.y=state.ball_vel.y;
     state.resort_friends();   //closeness of friends...
     state.resort_opps();      //...and opponents might have changed
     sum+=get_V_for_State(state);
     num++;
     }
     for(int i=0;i<100;i++)
     cerr<<"SUM: "<<sum<<" divide by: "<<num<<" gives AVERAGE OUTPUT of: "<<sum / num<<"\n\n";
     }
     //end calculate average output of net function
     */

    ///*
    //begin special test to give out net function as a 3D grid surface
    //3D grid surface for ball_holder
    if( ( WSinfo::me->number == 10 ) && ( WSinfo::ws->time == 1 ) && ( func_map_file[ 0 ] != '-' ) )
    {
        jkState state;
        //either from current world info:
        state.get_from_WS( WSinfo::get_teammate_by_number( WSmemory::teammate_last_at_ball_number ) );
        //or from a situation string:
        state.fromString( "30.0 0.0 0.0 0.0 0.0    30.5 0.0 0.0 0.0    50.0 0.0 0.0 0.0 0.0    40.0 -10.0 0.0 0.0 0.0    40.0 10.0 0.0 0.0 0.0    40.0 20.0 0.0 0.0 0.0    40.0 -20.0 0.0 0.0 0.0    30.0 -10.0 0.0 0.0 0.0    30.0 10.0 0.0 0.0 0.0" );

        FILE *griddata;
        strcat( func_map_file, ".bh" ); //bh ~ ball_holder
        griddata = fopen( func_map_file, "w" );
        if( !griddata )
        {
            cerr << "\n" << func_map_file
                    << " is not equal \"-\" but cannot be opened for writing either\nFATAL ERROR";
            exit( 999 );
        }
        double interval = 0.5;
        for( double v_x = 30.0; v_x <= 50.0; v_x += interval )
            for( double v_y = -20.0; v_y <= 20.0; v_y += interval )
            {
                state.my_pos.setX( v_x );
                state.my_pos.setY( v_y );
                state.resort_friends(); //closeness of friends...
                state.resort_opps();    //...and opponents might have changed
                ostringstream os;
                os << v_y << "  " << v_x << "  " << get_V_for_State( state )
                        << "\n" << ends;
                fprintf( griddata, "%s", os.str().c_str() );
            }
        fclose( griddata );
        func_map_file[ 0 ] = '-'; //be sure that we do it only once
    }

    //3D grid surface for non_ballholder
    if( ( WSinfo::me->number == 11 ) && ( WSinfo::ws->time == 1 ) && ( func_map_file[ 0 ] != '-' ) )
    {
        jkState state;
        //either from current world info:
        state.get_from_WS( WSinfo::get_teammate_by_number( WSmemory::teammate_last_at_ball_number ) );
        //or from a situation string:
        state.fromString( "30.0 0.0 0.0 0.0 0.0    30.5 0.0 0.0 0.0    50.0 0.0 0.0 0.0 0.0    40.0 -10.0 0.0 0.0 0.0    40.0 10.0 0.0 0.0 0.0    40.0 20.0 0.0 0.0 0.0    40.0 -20.0 0.0 0.0 0.0    30.0 -10.0 0.0 0.0 0.0    30.0 10.0 0.0 0.0 0.0" );

        FILE *griddata;
        strcat( func_map_file, ".nbh" ); //nbh ~ non-ball_holder
        griddata = fopen( func_map_file, "w" );
        if( !griddata )
        {
            cerr << "\n" << func_map_file
                    << " is not equal \"-\" but cannot be opened for writing either\nFATAL ERROR";
            exit( 999 );
        }
        double interval = 0.5;
        Vector fixed_friend_pos = state.friend_pos[ 0 ]; //this guys position is static
        for( double v_x = 30.0; v_x <= 50.0; v_x += interval )
            for( double v_y = -20.0; v_y <= 20.0; v_y += interval )
            {
                state.friend_pos[ 0 ].setX( fixed_friend_pos.getX() ); //all values need to be assigned again since order might be changed
                state.friend_pos[ 0 ].setY( fixed_friend_pos.getY() ); //due to varying closeness to ball_holder
                state.friend_pos[ 1 ].setX( v_x );
                state.friend_pos[ 1 ].setY( v_y );
                state.resort_friends();
                ostringstream os;
                os << v_y << "  " << v_x << "  " << get_V_for_State( state )
                        << "\n" << ends;
                fprintf( griddata, "%s", os.str().c_str() );
            }
        fclose( griddata );
        func_map_file[ 0 ] = '-'; //be sure that we do it only once
    }
    //end special test to give out net function as a 3D grid surface*/
    if( !use_me )
        return false;
    DBLOG_POL( 0, "SCORE04-HACK: At t=" << WSinfo::ws->time << ": score04tried=" << score04TriedCounter << "  score04executed=" << score04ExecutedCounter );
    score04TriedCounter++;
    //LOG_POL(0,<<"Score04 behavior called.");
    if( ( !train_mode ) && ( last_called != WSinfo::ws->time - 1 ) )
    {
        seq_started = WSinfo::ws->time;
        play_on_cnt = 0;
        reset_intention( false );
        /*if((train_mode)&&(WSinfo::me->number==10)){
         jkState state;
         state.get_from_WS();
         if((train_mode)&&(get_V_for_State(state))>REJECT_VAL){
         cerr << "\n\nREJECTED STARTSTATE\n\n";
         Intention intention;
         WSpset opp=WSinfo::alive_opponents;
         opp.keep_and_sort_closest_players_to_point(1,WSinfo::me->pos);
         Vector target=opp[0]->pos;
         float speed=Planning::compute_pass_speed(WSinfo::ball->pos,target,target);
         intention.set_pass(target,speed, WSinfo::ws->time, opp[0]->number, 0,
         target);
         jkwball03->my_blackboard.pass_or_dribble_intention = intention;
         jkwball03->my_blackboard.intention = intention;
         jkwball03->intention2cmd(jkwball03->my_blackboard.intention,cmd);
         return true;
         }
         }*/
    }
    last_called = WSinfo::ws->time;
    if( WSinfo::ws->play_mode == PM_PlayOn )
        play_on_cnt++;

    bool res = false;

    if( WSinfo::ws->play_mode == PM_PlayOn )
    {
        res = get_player_cmd( cmd );
        if( !res )
        {
            //don't return false and set a command!!!
            cmd.cmd_body.unset_lock();
            cmd.cmd_body.unset_cmd();
            return false;
        }
        else
            score04ExecutedCounter++;
    }
    return res;
}
//models somebodys try to intercept the ball - returns true if ball gets into the kickrange of this somebody
bool Score04::model_intercept( Vector my_pos, Vector my_vel, ANGLE my_ang,
        Vector ball_pos, Vector ball_vel, Vector &new_my_pos,
        Vector &new_my_vel, ANGLE &new_my_ang, Vector &new_ball_pos,
        Vector &new_ball_vel )
{
    Cmd sim;
    intercept->set_virtual_state( my_pos, my_vel, my_ang, ball_pos, ball_vel );
    if( intercept->get_cmd( sim ) )
    {
        Tools::model_cmd_main( my_pos, my_vel, my_ang, ball_pos, ball_vel, sim.cmd_body, new_my_pos, new_my_vel, new_my_ang, new_ball_pos, new_ball_vel );
    }
    else
    { //something went wrong - do nothing
        new_my_pos = my_pos + my_vel;
        new_my_vel = my_vel;
        new_my_ang = my_ang;
        new_ball_pos = ball_pos + ball_vel;
        new_ball_vel = ball_vel;
    }
    if( in_kickrange( new_my_pos, new_ball_pos ) )
        return true; //intercept successfull, ball in kickrange
    else
        return false;
}

void Score04::model_continue( Vector pos, Vector vel, Vector &new_pos,
        Vector &new_vel )
{
    new_pos = pos + vel;
    new_vel = vel * ServerOptions::player_decay;
}

void Score04::model_opp_continue( jkState state, int opp_num,
        Vector &new_my_pos, Vector &new_my_vel )
{
    new_my_pos = state.get_opp_pos( opp_num ) + state.opp_vel[ opp_num ];
    new_my_vel = state.opp_vel[ opp_num ] * ServerOptions::player_decay;
}

//ATTENTION: SHOULD LATERON USE THE REAL OPTIMISATION BEHAVIOUR OF FRIENDS!
//(might take too much time though...)
//friend_num -1 means model the my_XXX values
void Score04::model_friend_continue( jkState state, int friend_num,
        Vector &new_my_pos, Vector &new_my_vel )
{
    if( friend_num >= 0 )
    {
        new_my_pos = state.get_friend_pos( friend_num ) + state.friend_vel[ friend_num ];
        new_my_vel = state.friend_vel[ friend_num ] * ServerOptions::player_decay;
    }
    else
    {
        new_my_pos = state.my_pos + state.my_vel;
        new_my_vel = state.my_vel * ServerOptions::player_decay;
    }
}

void Score04::model_goalie( Vector my_pos, Vector my_vel, ANGLE my_ang,
        Vector ball_pos, Vector ball_vel, Vector &new_my_pos,
        Vector &new_my_vel, ANGLE &new_my_ang )
{
    //goalie will probably try to get onto shortest line between ball and goal
    //goalie can move (0.0/[0.0-1.0]), i.e. only on y axis
    Vector res;
    Line2d goalline( Vector( 52.0, 0.0 ), Vector( 0.0, 1.0 ) );
    Vector ball_on_goalline;
    Geometry2d::projection_to_line( ball_on_goalline, ball_pos, goalline );

    //bring in goalrange (since line is endless point might be wherever)
    if( ball_on_goalline.getY() > 7.5 )
        ball_on_goalline.setY( 7.5 );
    if( ball_on_goalline.getY() < -7.5 )
        ball_on_goalline.setY( -7.5 );

    Line2d ball_goal( ball_pos, ball_on_goalline );
    Vector target;
    Geometry2d::projection_to_line( target, my_pos, ball_goal );
    target -= my_pos;
    if( target.norm() > 1.0 ) //MaxPlayerMovement = 1.0
        target.normalize( 1.0 );
    new_my_pos = my_pos + target;
    new_my_vel = target;
    new_my_ang = my_ang;
}

//helper to get the angles to the next state since they are mostly not predicted
void copy_angles( jkState state, jkState &resulting_state )
{
    resulting_state.my_ang = state.my_ang;
    for( int i = 0; i < number_of_defenders; i++ )
        resulting_state.opp_ang[ i ] = state.opp_ang[ i ];
    for( int i = 0; i < number_of_attackers; i++ )
        resulting_state.friend_ang[ i ] = state.friend_ang[ i ];
}

//simulate playing pass to target (/to teammates pos), returns number of steps used
//until catched by teammate (using intercept target) or an error value otherwise
//(since recursive: ball_shot to signal that the ball was kicked already; set initially to zero)
//count to prevent too many loops
int Score04::model_pass( jkState state, jkState &resulting_state, Vector target,
        int pass_shot, int count, int player )
{
    Cmd sim;
    double speed;
    bool sim_cmd_set;
    jkState tmp_state;
    if( count >= MAX_PREDICTION_DEPTH )
    { // otherwise prediction might take us so long that we miss a cycle!
        state.copy_to( resulting_state );
        Vector my_pos2 = resulting_state.my_pos;
        Vector my_vel2 = resulting_state.my_vel;
        ANGLE my_ang2 = resulting_state.my_ang;
        resulting_state.my_pos = resulting_state.get_friend_pos( player );
        resulting_state.my_vel = resulting_state.friend_vel[ player ];
        resulting_state.my_ang = resulting_state.friend_ang[ player ];
        resulting_state.friend_pos[ player ] = my_pos2;
        resulting_state.friend_vel[ player ] = my_vel2;
        resulting_state.friend_ang[ player ] = my_ang2;
        resulting_state.ball_pos = resulting_state.my_pos;
        resulting_state.ball_vel = Vector( 0.0, 0.0 );
        resulting_state.resort_opps();
        PlayerSet dangers = WSinfo::alive_opponents;
        Vector elongation = target - state.ball_pos;
        elongation.normalize( 2.0 );
        dangers.keep_players_in_quadrangle( state.ball_pos, target + elongation, max( 1.0, min( state.ball_pos.distance( WSinfo::me->pos ), 4.5 ) ), WSinfo::me->pos.distance( target ) / 1.7 );
#ifdef DEBUG
        if( dangers.num >= 1 )
        {
            DBLOG_POL( LOG_LEVEL, "PLAYPASS MAX_PREDICTION_DEPTH ERREICHT, REJECT" )
        }
        else
        {
            DBLOG_POL( LOG_LEVEL, "PLAYPASS MAX_PREDICTION_DEPTH ERREICHT, IS GOOD" );
        }
        DBLOG_DRAW( LOG_LEVEL, Quadrangle2d( state.ball_pos, target + elongation, max( 1.0, min( state.ball_pos.distance( WSinfo::me->pos ), 3.0 ) ), WSinfo::me->pos.distance( target ) / 1.7 ) );
#endif
        if( dangers.num >= 1 )
        { // rough estimate: if enemy too close to passway better reject the action
            return LOOSE_SITUATION;
        }
        else
        {
            return count;
        }
    }
    if( pass_shot == 0 )
    { // model ball and the my_XXX behaviour
        neurokick->set_state( state.my_pos, state.my_vel, state.my_ang, state.ball_pos, state.ball_vel );
        speed = Planning::compute_pass_speed( state.ball_pos, target, target );
        neurokick->kick_to_pos_with_initial_vel( speed, target );
        sim_cmd_set = neurokick->get_cmd( sim );
        neurokick->reset_state();
        if( sim_cmd_set )
        {
            Tools::model_cmd_main( state.my_pos, state.my_vel, state.my_ang, state.ball_pos, state.ball_vel, sim.cmd_body, tmp_state.my_pos, tmp_state.my_vel, tmp_state.my_ang, tmp_state.ball_pos, tmp_state.ball_vel );
            if( !in_kickrange( tmp_state.my_pos, tmp_state.ball_pos ) )
                pass_shot++; //ball has left my kickable area, don't try to kick again
        }
        else
            return CMD_NOT_SET;
    }
    else
    { // ball was already kicked
        pass_shot++;
        model_friend_continue( state, -1, tmp_state.my_pos, tmp_state.my_vel );
        model_continue( state.ball_pos, state.ball_vel, tmp_state.ball_pos, tmp_state.ball_vel );
    } // end model ball and the my_XXX behaviour

    int intercepting_friend_num;
    int continuing_friend_num;
    Vector dummy1, dummy2;
    // begin model friends
    if( state.friend_pos[ 0 ].sqr_distance( target ) < state.friend_pos[ 1 ].sqr_distance( target ) )
    {
        //closest teammate should go to the ball
        intercepting_friend_num = 0;
        continuing_friend_num = 1;
    }
    else
    {
        //2nd closest teammate should go to the ball
        intercepting_friend_num = 1;
        continuing_friend_num = 0;
    }
    model_intercept( state.get_friend_pos( intercepting_friend_num ), state.friend_vel[ intercepting_friend_num ], state.friend_ang[ intercepting_friend_num ], target, Vector( .0, .0 ), tmp_state.friend_pos[ intercepting_friend_num ], tmp_state.friend_vel[ intercepting_friend_num ], tmp_state.friend_ang[ intercepting_friend_num ], dummy1, dummy2 );
    model_friend_continue( state, continuing_friend_num, tmp_state.friend_pos[ continuing_friend_num ], tmp_state.friend_vel[ continuing_friend_num ] );
    //end model friends
    //begin model opps
    int intercepting_opp_num = get_closest_opp_to( state, state.ball_pos );
    for( int i = 0; i < number_of_defenders; i++ )
    {
        if( intercepting_opp_num == i )
            model_intercept( state.get_opp_pos( intercepting_opp_num ), state.opp_vel[ intercepting_opp_num ], state.opp_ang[ intercepting_opp_num ], state.ball_pos, state.ball_vel, tmp_state.opp_pos[ intercepting_opp_num ], tmp_state.opp_vel[ intercepting_opp_num ], tmp_state.opp_ang[ intercepting_opp_num ], dummy1, dummy2 );
        else if( i == 0 )
            model_goalie( state.opp_pos[ 0 ], state.opp_vel[ 0 ], state.opp_ang[ 0 ], state.ball_pos, state.ball_vel, tmp_state.opp_pos[ 0 ], tmp_state.opp_vel[ 0 ], tmp_state.opp_ang[ 0 ] );
        else
            model_opp_continue( state, i, tmp_state.opp_pos[ i ], tmp_state.opp_vel[ i ] );
    }
    //end model opps
    //check pass success
    for( int i = 0; i < number_of_attackers; i++ )
    {
        if( in_kickrange( tmp_state.friend_pos[ i ], tmp_state.ball_pos ) )
        {
            tmp_state.copy_to( resulting_state );
            return count;
        }
    }
    //check opp intercept success
    for( int i = 0; i < number_of_defenders; i++ )
    {
        if( in_kickrange( tmp_state.opp_pos[ i ], tmp_state.ball_pos ) )
        {
            tmp_state.copy_to( resulting_state );
            return LOOSE_SITUATION;
        }
    }
    //OK not finished yet, go on recursively
    if( use_model_ballholder )
    {
        return model_pass( tmp_state, resulting_state, target, pass_shot, ++count, player );
    }
    else
    { //do not predict recursively
        tmp_state.copy_to( resulting_state );
        return count;
    }
}

bool Score04::default_model( Cmd &sim, jkState state, jkState &resulting_state )
{
    bool loss_state = false;
    Vector temp_ball_pos, temp_ball_vel;
    Tools::model_cmd_main( state.my_pos, state.my_vel, state.my_ang, state.ball_pos, state.ball_vel, sim.cmd_body, resulting_state.my_pos, resulting_state.my_vel, resulting_state.my_ang, resulting_state.ball_pos, resulting_state.ball_vel );
    //model goalie
    model_goalie( state.opp_pos[ 0 ], state.opp_vel[ 0 ], state.opp_ang[ 0 ], state.ball_pos, state.ball_vel, resulting_state.opp_pos[ 0 ], resulting_state.opp_vel[ 0 ], resulting_state.opp_ang[ 0 ] );
    //model closest opponents try to get the ball
    if( model_intercept( state.opp_pos[ 1 ], state.opp_vel[ 1 ], state.opp_ang[ 1 ], state.ball_pos, state.ball_vel, resulting_state.opp_pos[ 1 ], resulting_state.opp_vel[ 1 ], resulting_state.opp_ang[ 1 ], temp_ball_pos, temp_ball_vel ) )
    {
        //uh oh, opponent got me
        loss_state = true;
    } //else forget about the new temp ball_pos and -vel
      //model not so important actors (valid assumption?)
    for( int i = 2; i < number_of_defenders; i++ )
        model_opp_continue( state, i, resulting_state.opp_pos[ i ], resulting_state.opp_vel[ i ] );
    for( int i = 0; i < number_of_attackers; i++ )
        model_friend_continue( state, i, resulting_state.friend_pos[ i ], resulting_state.friend_vel[ i ] );
    return loss_state;
}

bool Score04::model_turn_and_dash( Cmd &cmd, jkState state )
{
    // used for turn_and_dash
    Vector dummy1, dummy3;
    Vector mynewpos, ballnewpos;
    double dummy2;
    int dash = 0;
    Cmd_Body testcmd;
    double required_turn;
    bool dash_found;

    required_turn = Tools::get_angle_between_null_2PI( state.ball_vel.arg() - state.my_ang.get_value() );
    // If I can get ball by dashing, dash
    dash_found = false;
    if( Tools::get_abs_angle( required_turn ) < 40. / 180. * PI )
    {
        for( dash = 100; dash >= 30; dash -= 10 )
        {
            testcmd.unset_lock();
            testcmd.unset_cmd();
            testcmd.set_dash( dash );
            Tools::model_cmd_main( state.my_pos, state.my_vel, state.my_ang.get_value(), state.ball_pos, state.ball_vel, testcmd, mynewpos, dummy1, dummy2, ballnewpos, dummy3 );
            if( ( mynewpos - ballnewpos ).norm() < 0.8 * ServerOptions::kickable_area )
            {
                dash_found = true;
                break;
            }
        }
    }
    if( dash_found == true )
    {
        basiccmd->set_dash( dash );
        basiccmd->get_cmd( cmd );
        return true;
    }

    if( Tools::get_abs_angle( required_turn ) > 10. / 180. * PI )
    {
        basiccmd->set_turn_inertia( required_turn );
        basiccmd->get_cmd( cmd );
        return true;
    }
    // dash forward
    basiccmd->set_dash( 100 );
    basiccmd->get_cmd( cmd );
    return true;
}

bool Score04::is_loose_situation( jkState state )
{
    if( fabs( state.ball_pos.getX() ) > ServerOptions::pitch_length / 2 )
        return true;
    if( fabs( state.ball_pos.getX() ) < 5.0 ) //too close to other field players
        return true;
    if( fabs( state.ball_pos.getY() ) > ServerOptions::pitch_width / 2 )
        return true;
    for( int i = 0; i < number_of_defenders; i++ )
        if( in_kickrange( state.opp_pos[ i ], state.ball_pos ) )
            return true;
    return false;
}

int Score04::model_selfpass( jkState state, jkState &resulting_state,
        double speed, Vector target, ANGLE ang )
{
    Cmd sim;
    Cmd sim2;
    Cmd sim3;
    Cmd sim4;
    bool sim_cmd_set;
    int cycles_needed;
    double speed1, speed2;
    //onetwokick->reset_state(); // use current ws-state
    onetwokick->set_state( state.my_pos, state.my_vel, state.my_ang, state.ball_pos, state.ball_vel, state.opp_pos[ 1 ], state.opp_ang[ 1 ], 0 );
    onetwokick->kick_to_pos_with_initial_vel( speed, target );
    onetwokick->get_vel( speed1, speed2 );
    if( fabs( speed1 - speed ) > 0.1 )
    {
        cycles_needed = 2;
    }
    else
        cycles_needed = 1;
    jkState tmp_state;
    sim_cmd_set = selfpass->get_cmd( state.my_pos, state.my_vel, state.my_ang, state.ball_pos, state.ball_vel, state.opp_pos[ 1 ], state.opp_ang[ 1 ], sim, ang, speed, target );
    if( sim_cmd_set )
    {
        if( default_model( sim2, state, tmp_state ) )
            return LOOSE_SITUATION; //opp got me by intercepting
    }
    else
        return CMD_NOT_SET;
    if( ( cycles_needed == 1 ) || ( !use_model_ballholder ) )
    {
        tmp_state.copy_to( resulting_state );
        if( !use_model_ballholder ) //we came here since modelling over several cycles has been turned off
            cycles_needed = 1; //in the config file so we only predict one cycle - prevents model inaccurcies
    }
    else
    { //but only very small differences in succ. states might be misleading, too
        sim_cmd_set = selfpass->get_cmd( tmp_state.my_pos, tmp_state.my_vel, tmp_state.my_ang, tmp_state.ball_pos, tmp_state.ball_vel, tmp_state.opp_pos[ 1 ], tmp_state.opp_ang[ 1 ], sim3, ang, speed, target );
        if( sim_cmd_set )
        {
            if( default_model( sim4, tmp_state, resulting_state ) )
                return LOOSE_SITUATION; //opp got me by intercepting
        }
        else
            return CMD_NOT_SET;
    }
    /*advance me to target pos, not used since usually this position is not reached!
     resulting_state.my_pos.x=target.x;
     resulting_state.my_pos.y=target.y;
     resulting_state.ball_pos.x=target.x;
     resulting_state.ball_pos.y=target.y;
     */
    return cycles_needed;
}

//given an action and a (current) state this predicts the expected state
//after applying the action - even if it takes some cycles; returns number of cycles used for the action
//or an error value if something goes wrong
int Score04::predict_outcome( jkState state, Score04Action *actions, int which,
        jkState &resulting_state )
{
    Cmd sim;
    Cmd sim2;
    Cmd sim3;
    Cmd sim4;
    bool sim_cmd_set;
    copy_angles( state, resulting_state );
    switch( actions[ which ].type )
    {
        case 1 :
        case 2 :
        { //pass to teammate
            Vector target;
            if( actions[ which ].param < 0.5 )
            { //determine target
                target = state.get_friend_pos( actions[ which ].type - 1 ) + state.friend_vel[ actions[ which ].type - 1 ];
            }
            else
                //just take given target, e.g. for laufpasses
                target = actions[ which ].target;
            //begin filter some circumstances where I got no chance in order to save calculation time
            if( state.my_pos.distance( target ) > 20.0 )
                return LOOSE_SITUATION;
            //end filter some circumstances where I got no chance in order to save calculation time
            return model_pass( state, resulting_state, target, 0, 0, actions[ which ].type - 1 );
            break;
        }
        case 3 :
        { //dribble
          //model ballholders action (dribbling)
            dribblestraight->last_calc = 0; //reset dribbles' cache
            MyState tmp_state = state.get_old_version_State();
            sim_cmd_set = dribblestraight->calc_next_cmd( sim, tmp_state );
            if( sim_cmd_set )
            {
                if( default_model( sim2, state, resulting_state ) )
                    return LOOSE_SITUATION; //opp got me by intercepting
            }
            else
                return CMD_NOT_SET;
            return 1; //is always one turn
            break;
        }
        case 4 :
        { //immediate_selfpass
            Vector target;
            double speed, kickdir;
            kickdir = actions[ which ].param;
            target.init_polar( 1., kickdir );
            target += WSinfo::ball->pos;
            speed = 0.9; //Planning::compute_pass_speed(WSinfo::ball->pos,target,target);
            if( ( Tools::get_abs_angle( state.ball_vel.arg() - kickdir ) < 10 / 180. * PI ) && fabs( state.ball_vel.norm() - speed ) < 0.05 )
            {
                // Ball already has desired dir and speed
                sim_cmd_set = model_turn_and_dash( sim, state );
                if( sim_cmd_set )
                {
                    if( default_model( sim2, state, resulting_state ) )
                        return LOOSE_SITUATION; //opp got me by intercepting
                }
                else
                    return CMD_NOT_SET;
            }
            else
            {
                neurokick->set_state( state.my_pos, state.my_vel, state.my_ang, state.ball_pos, state.ball_vel );
                neurokick->kick_to_pos_with_initial_vel( speed, target );
                sim_cmd_set = neurokick->get_cmd( sim );
                neurokick->reset_state();
                if( sim_cmd_set )
                {
                    if( default_model( sim2, state, resulting_state ) )
                        return LOOSE_SITUATION; //opp got me by intercepting
                }
                else
                    return CMD_NOT_SET;
            }
            return 1; //is always one turn
            break;
        }
        case 5 :
        { //selfpass
            double speed = actions[ which ].param2;
            Vector target = actions[ which ].target;
            ANGLE ang = ANGLE( actions[ which ].param );
            return model_selfpass( state, resulting_state, speed, target, ang );
            break;
        }
        default :
            cerr << "Unknown action - something really went wrong!\n";
            return LOOSE_SITUATION;
    }
}

//selects the action with the least expected costs
//returns -1 if none good
int Score04::select_action( Score04Action *actions, int num, jkState &state,
        float &bestval )
{
    jkState resulting_state;
    eval_actions = new Sort( num );
    int needed_cycles;
    int which;
    float val;
    eval_action_cnt = 0;
    for( int i = 0; i < num; i++ )
    {
        needed_cycles = predict_outcome( state, actions, i, resulting_state );
        if( ( needed_cycles == CMD_NOT_SET ) || ( needed_cycles >= LOOSE_SITUATION ) )
        {
            //no cmd will be available or it will lead to a loose, skip this action
        }
        else if( needed_cycles < 99 )
        {
            //feed net...
            val = get_V_for_State( resulting_state ); //get predicted costs for resulting state
            if( cost_per_action )
                val = val + ( 1 * immediate_cost ); //add immediate costs of action, here 1 per action indifferent of its length
            else
                val = val + ( needed_cycles * immediate_cost ); //add immediate costs of action
            eval_actions->add( i, val );
            eval_action_cnt++;
#ifdef DEBUG
            DBLOG_POL( LOG_LEVEL, "action " << i << " of type " << actions[i].type << " got value " << val );
#endif
        }
    }
    eval_actions->do_sort(); //sort; smallest/cheapest first
    //begin randomize between best actions if there are several with the same value
    int ix = 0;
    val = eval_actions->get_value( 0 );
    while( ( eval_action_cnt > ix + 1 ) && ( eval_actions->get_value( ix + 1 ) == val ) )
        ix++;
    ix = rand_in_range( 0, ix );
    //end randomize between best actions if there are several with the same value
    which = eval_actions->get_key( ix );
    val = eval_actions->get_value( ix );
#ifdef DEBUG
    if( val != -1 )
    {
        DBLOG_POL( LOG_LEVEL, "least cost action is of type " << actions[which].type << " and got value " << val );
    }
#endif
    bestval = val;

    return which;
}

void set_pass_msg( Vector target )
{
    WSinfo::jk_pass_msg_set = true;
    if( ( target.getX() > 0.0 ) && ( target.getX() < 0.1 ) ) //very small values cause errors, simple workaround
        target.setX( 0.0 );
    if( ( target.getX() < 0.0 ) && ( target.getX() > -0.1 ) )
        target.setX( 0.0 );
    if( ( target.getY() > 0.0 ) && ( target.getY() < 0.1 ) )
        target.setY( 0.0 );
    if( ( target.getY() < 0.0 ) && ( target.getY() > -0.1 ) )
        target.setY( 0.0 );
    char *buffer;
    char res[ 8 ];
    char tmp[ 8 ];
    int precision = 3;
    int decimal, sign, offset;
    buffer = ecvt( target.getX(), precision, &decimal, &sign );
    if( sign )
    {
        strcpy( res, "-" );
        strcat( res, buffer );
        offset = 1;
    }
    else
    {
        strcpy( res, buffer );
        offset = 0;
    }
    strcpy( tmp, res + offset + decimal );
    strcpy( res + offset + decimal + 1, tmp );
    res[ decimal + offset ] = '.';
    res[ 4 ] = '\0';
    strcpy( WSinfo::jk_pass_msg, res );

    buffer = ecvt( target.getY(), precision, &decimal, &sign );
    if( sign )
    {
        strcpy( res, "-" );
        strcat( res, buffer );
        offset = 1;
    }
    else
    {
        strcpy( res, buffer );
        offset = 0;
    }
    strcpy( tmp, res + offset + decimal );
    strcpy( res + offset + decimal + 1, tmp );
    res[ decimal + offset ] = '.';
    res[ 4 ] = '\0';
    strcat( WSinfo::jk_pass_msg, res );
#ifdef DEBUG
    DBLOG_POL( LOG_LEVEL, "SAY PASS TARGET: " << WSinfo::jk_pass_msg );
#endif
}

//maps the selected action to an intention
bool Score04::apply_action( Score04Action &do_it, Intention &intention )
{
    Vector target;
    double speed, kickdir;
    last_action_pass = false;
    last_action_type = do_it.type;
    last_action_time = WSinfo::ws->time;
    switch( do_it.type )
    {
        case 1 : //pass to a teammate
        case 2 :
        {
            PlayerSet team = WSinfo::alive_teammates_without_me;
            team.keep_and_sort_closest_players_to_point( number_of_attackers, WSinfo::me->pos );
            int i = do_it.type - 1; //0=closest player 1=2nd closest player
            target = team[ i ]->pos;
            last_action_pass_target_num = team[ i ]->number;
            speed = Planning::compute_pass_speed( WSinfo::ball->pos, target, target );
            if( WSinfo::me->pos.distance( target ) < 11 )
                speed -= speed * 0.1;

#ifdef DEBUG
            DBLOG_POL( LOG_LEVEL, "applying action pass; target (" << target.getX () << "/" << target.getY () << ") with speed" << speed );
            DBLOG_DRAW( LOG_LEVEL, VC2D (target, 0.3, "red") );
#endif

            intention.set_pass( target, speed, WSinfo::ws->time, team[ i ]->number, 0, target );
            last_action_pass = true;
            last_action_pass_time = WSinfo::ws->time;
            //JK PASS_MSG_HACK begin
            if( WSinfo::me_full != NULL )
            {
                set_pass_msg( target );
            }
            //JK PASS_MSG_HACK end
            return true;
            break;
        }
        case 3 :
        { //dribble
#ifdef DEBUG
            DBLOG_POL( LOG_LEVEL, "applying action set_dribble" );
#endif
            intention.set_dribble( Vector( 52.0, 0.0 ), WSinfo::ws->time );
            return true;
            break;
        }
        case 4 :
        {
            kickdir = do_it.param;
            target.init_polar( 1., kickdir );
            target += WSinfo::ball->pos;
            speed = Planning::compute_pass_speed( WSinfo::ball->pos, target, target );
#ifdef DEBUG
            DBLOG_POL( LOG_LEVEL, "applying action immediate_self_pass; target (" << target.getX () << "/" << target.getY () << ") with speed" << speed );
            DBLOG_DRAW( LOG_LEVEL, VC2D (target, 0.3, "red") );
#endif
            intention.set_immediateselfpass( target, speed, WSinfo::ws->time );
            //set_neck_selfpass ?
            return true;
            break;
        }
        case 5 :
        {
            kickdir = do_it.param;
            target = do_it.target;
            speed = do_it.param2;
#ifdef DEBUG
            DBLOG_POL( LOG_LEVEL, "applying action self_pass; target (" << target.getX () << "/" << target.getY () << ") with speed" << speed );
            DBLOG_DRAW( LOG_LEVEL, VC2D (target, 0.3, "red") );
#endif
            intention.set_selfpass( ANGLE( kickdir ), target, speed, WSinfo::ws->time );
            //set_neck_selfpass ?
            return true;
            break;
        }
        default :
            return false;
    }
}

bool in_field( Vector po )
{
    if( ( fabs( po.getX() ) < FIELD_BORDER_X ) && ( fabs( po.getY() ) < FIELD_BORDER_Y ) )
        return true;
    return false;
}

bool in_range_pos( Vector po )
{
    if( ( po.getX() < WSinfo::his_team_pos_of_offside_line() - 1.0 ) && ( po.getX() > FIELD_BORDER_X - 25.0 ) )
        return true;
    return false;
}

bool Score04::get_player_cmd( Cmd &cmd )
{
    Vector target_pos;
#ifdef DEBUG
    DBLOG_POL( LOG_LEVEL, "Action_Memory: (" << last_action_time << "/" << last_action_type << ")" << " , MSG_Memory: " << WSinfo::jk_pass_msg );
#endif
    int min_dist = ServerOptions::kickable_area;
    bool target_pos_set = false;
#if LOGGING && BASIC_LOGGING
    long starttime = Tools::get_current_ms_time();
#endif
    //JK PASS_MSG_HACK begin
    if( WSinfo::me_full != NULL )
    {
        WSinfo::jk_pass_msg_set = false;
        //if((last_action_time>WSinfo::ws->time-2)&&((last_action_type==1)||(last_action_type==2)))
        //  WSinfo::jk_pass_msg_rec_time=-99;  //it has been me shooting the pass so ignore
        //WSpset ball_area=WSinfo::valid_teammates;
        //ball_area.keep_players_in_circle(WSinfo::ball->pos,2.0);
        target_pos.setXY( WSinfo::jk_pass_msg_x, WSinfo::jk_pass_msg_y );
        target_pos_set = true;
        PlayerSet closeness = WSinfo::valid_teammates;
        closeness.keep_and_sort_closest_players_to_point( 1, target_pos );
        if( ( closeness[ 0 ]->number == WSinfo::me->number /*target_pos.distance(WSinfo::me->pos)<2.0*/) && ( WSinfo::jk_pass_msg_rec_time > WSinfo::ws->time - 5 ) && ( WSinfo::ball->pos.distance( WSinfo::me->pos ) > min_dist ) )
        {
#ifdef DEBUG
            DBLOG_POL( LOG_LEVEL, "still PASS_INFO: (" << target_pos.getX () << "/" << target_pos.getY () << ")" );
#endif
            Vector future_vel( WSinfo::ball->vel * ServerOptions::ball_decay );
            //if(WSinfo::me->pos.distance(target_pos)>0) //when ball comes close use real data
            //intercept->set_virtual_state(WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang,target_pos,future_vel);
            return intercept->get_cmd( cmd );
        }
        else
        {
            WSinfo::jk_pass_msg_rec_time = -99;
        }
        if( WSinfo::jk_pass_msg_rec )
        {
            WSinfo::jk_pass_msg_rec_time = WSinfo::ws->time;
            WSinfo::jk_pass_msg_rec = false;
            char res[ 80 ];
            char fin1[ 80 ];
            char fin2[ 80 ];
            strcpy( res, WSinfo::jk_pass_msg );
            char *pos1 = strchr( res, '"' );
            if( pos1 == NULL )
                return 1;
            char *pos2 = strchr( pos1 + 1, '"' );
            if( pos2 == NULL )
                return 1;
            pos2[ 0 ] = '\0';
            strcpy( fin2, pos1 + 5 );
            pos2 = pos1 + 5;
            pos2[ 0 ] = '\0';
            strcpy( fin1, pos1 + 1 );
            target_pos.setXY( atof( fin1 ), atof( fin2 ) );
            target_pos_set = true;
            PlayerSet closeness2 = WSinfo::valid_teammates;
            closeness2.keep_and_sort_closest_players_to_point( 1, target_pos );
            WSinfo::jk_pass_msg_x = target_pos.getX();
            WSinfo::jk_pass_msg_y = target_pos.getY();
            if( ( closeness2[ 0 ]->number == WSinfo::me->number ) /*(target_pos.distance(WSinfo::me->pos)<1.1)*/&& ( WSinfo::ball->pos.distance( WSinfo::me->pos ) > min_dist ) )
            {
#ifdef DEBUG
                DBLOG_POL( LOG_LEVEL, "got PASS_INFO: (" << target_pos.getX () << "/" << target_pos.getY () << ")" );
#endif
                Vector future_vel( WSinfo::ball->vel * ServerOptions::ball_decay );
                //intercept->set_virtual_state(WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang,target_pos,future_vel);
                return intercept->get_cmd( cmd );
            }
        }
    }
    //JK PASS_MSG_HACK end
    bool result;
    bool new_intention = true;
    int i;
    if( train_mode )
    { //special modes for training (have to be set like that since lots of errors with other methods):
      //new tajectory, reset global variables
        if( WSinfo::me->pos.getX() < -10.0 && WSinfo::me->pos.getX() > -19.0 )
        {
            reset_intention( true );
        }
        //protocol stored data of all recent trajectories to file
        if( WSinfo::me->pos.getX() < -20.0 && WSinfo::me->pos.getX() > -29.0 )
        {
            protocol();
            exit( 0 );
        }
    }
    ball_holder = WSmemory::teammate_last_at_ball_number;

    if( ( last_action_time > WSinfo::ws->time - 4 ) && ( ( last_action_type == 1 ) || ( last_action_type == 2 ) ) )
    {
        last_action_pass = true;
        ball_holder = last_action_pass_target_num;
    }
    else
        last_action_pass = false;

    if( ( !last_action_pass ) && !WSinfo::is_ball_kickable() && ( ball_holder == WSinfo::me->number ) )
    {
//ball lost out of kickrange by chance, use default behaviour to get it back
#ifdef DEBUG
        DBLOG_POL( LOG_LEVEL, "ball lost out of kickrange by chance, use default behaviour to get it back" );
#endif
        return false;
    }
    result = false;

    if( !train_mode )
        if( !WSinfo::is_ball_pos_valid() )
            return false;

    if( !WSinfo::is_ball_kickable() )
    { //&&((ball_holder!=WSinfo::me->number)/*||(WSinfo::ball->pos.distance(WSinfo::me->pos)>3&&(WSinfo::ball->pos.distance(WSinfo::me->pos)<WSinfo::me->pos.distance(WSinfo::ball->pos+WSinfo::ball->vel)))*/)){
//I am a non-ball-holder
#ifdef DEBUG
        DBLOG_POL( LOG_LEVEL, "not got Ball, in Score04" );
#endif
        holding_ball_for = 0;
        last_teammate_excl_me_at_ball = WSmemory::teammate_last_at_ball_number;
        if( !use_me_nonballholder ) //use default behaviour for positioning
            return false;
        if( WSinfo::me->pos.distance( WSinfo::ball->pos ) > 20.0 ) //Too far away from ball for the net
            return false;
        PlayerSet team = WSinfo::alive_teammates;
        if( ( target_pos_set ) && ( ( last_action_type == 1 ) || ( last_action_type == 2 ) ) && ( last_action_time > WSinfo::ws->time - 3 ) )
            team.keep_and_sort_closest_players_to_point( 2, target_pos ); //ball is on its way to this place
        else
            team.keep_and_sort_closest_players_to_point( 2, WSinfo::ball->pos );
        if( ( team[ 0 ] != NULL ) && ( in_kickrange( team[ 0 ]->pos, WSinfo::ball->pos ) ) )
            ball_holder = team[ 0 ]->number;
        PPlayer bholder = WSinfo::get_teammate_by_number( ball_holder );
        team.remove( WSinfo::get_teammate_by_number( WSmemory::teammate_last_at_ball_number ) );
        if( ( ( ( bholder != NULL ) && ( !WSinfo::is_ball_kickable_for( bholder ) ) && team[ 0 ]->number == WSinfo::me->number ) || ( ( WSinfo::valid_teammates.get_player_with_newest_pass_info() != NULL ) && ( WSinfo::me->number == WSinfo::valid_teammates.get_player_with_newest_pass_info()->number ) ) ) /*||(last_action_type>2)*/)
        { //ball was shot by holder (!=me) and I am closest -> get it
            ball_holder = -1;
#ifdef DEBUG
            DBLOG_POL( LOG_LEVEL, "\n" << WSinfo::me->number << ": Ball shot by holder and I am closest -> trying to get it." );
#endif
            //return intercept->get_cmd(cmd);
            return false; //use default behaviour to get ball
        }
        else
        {
            if( bholder == NULL )
            {
                ball_holder = -1;
#ifdef DEBUG
                cerr << "\n" << WSinfo::me->number
                        << ": Hey this should not happen! (fatal err 1).\n";
#endif
                return false;
            }
            //freilaufen, e.g. optimize with samples over value function

            if( cached_no_ball_pos.getY() > 0 )
            { //we still have a target to reach
#ifdef DEBUG
                DBLOG_POL( LOG_LEVEL, "we still have a target to reach\n" );
#endif
                go2pos->set_target( cached_no_ball_pos, .1, 0, 0 ); //jk was before: go2pos->set_target(possible_pos[which],.5,1,true);
                cached_no_ball_pos.setY( -1 );
                if( go2pos->get_cmd( cmd ) )
                {
                    return true;
                } //otherwise we are there already and select a new place to go
                else
                {
#ifdef DEBUG
                    DBLOG_POL( LOG_LEVEL, "NO, we are there already\n" );
#endif
                }
            }
            if( /*(last_action_type==1)||(last_action_type==2)||*/( ( bholder != NULL ) && ( !WSinfo::is_ball_kickable_for( bholder ) ) && ( WSinfo::ball->pos.distance( bholder->pos ) > 4 ) ) )
            {
                bholder = WSinfo::get_teammate_by_number( team[ 0 ]->number );
                if( team[ 0 ]->number == WSinfo::me->number ) //should not happen
                    return false;
#ifdef DEBUG
                DBLOG_POL( LOG_LEVEL, "simulate that ball was already passed to " << team[0]->number );
#endif
            } //simulate that ball was already passed and try to get in a good position for the ballholder-to-be
            jkState state;
            state.get_from_WS( bholder );
            int resolution = 18; //every 20 degrees
            Vector *possible_pos = new Vector[ 2 * resolution + 1 ];
            int posi_count = 0;
            int posi_count2 = 0;
            for( i = 0; i < resolution; i++ )
            {
                possible_pos[ posi_count ] = Vector( ANGLE( 2 * M_PI / resolution * i ) );
                possible_pos[ posi_count ].normalize( 1. );
                possible_pos[ posi_count++ ] += WSinfo::me->pos;
                possible_pos[ posi_count ] = Vector( ANGLE( 2 * M_PI / resolution * i ) );
                possible_pos[ posi_count ].normalize( 2. );
                possible_pos[ posi_count++ ] += WSinfo::me->pos;
            }
            possible_pos[ posi_count++ ] = WSinfo::me->pos; //last_good_pos;
            Sort *posis = new Sort( posi_count );
            PlayerSet team = WSinfo::alive_teammates;
            team.keep_and_sort_closest_players_to_point( 3, bholder->pos );
            int my_friend_num = -1;
            for( i = 1; i < 3; i++ )
                if( ( team[ i ] != NULL ) && ( team[ i ]->number == WSinfo::me->number ) )
                    my_friend_num = i - 1;
            if( my_friend_num == -1 ) // I am further away than the important others
                result = false;
            //maybe state should be advanced here to increase prediction accuracy...
            if( use_model_nonballholder )
            { //use model to determine where other players will go in next state
                jkState new_state;
                copy_angles( state, new_state );
                model_goalie( state.opp_pos[ 0 ], state.opp_vel[ 0 ], state.opp_ang[ 0 ], state.ball_pos, state.ball_vel, new_state.opp_pos[ 0 ], new_state.opp_vel[ 0 ], new_state.opp_ang[ 0 ] );
                for( i = 1; i < number_of_defenders; i++ )
                    model_continue( state.opp_pos[ i ], state.opp_vel[ i ], new_state.opp_pos[ i ], new_state.opp_vel[ i ] );
                for( i = 0; i < number_of_attackers; i++ )
                    model_continue( state.friend_pos[ i ], state.friend_vel[ i ], new_state.friend_pos[ i ], new_state.friend_vel[ i ] );
                model_continue( state.my_pos, state.my_vel, new_state.my_pos, new_state.my_vel );
                model_continue( state.ball_pos, state.ball_vel, new_state.ball_pos, new_state.ball_vel );
                double temp_V;
                for( int j = 0; j < posi_count; j++ )
                {
                    new_state.set_friend_pos( my_friend_num, possible_pos[ j ] );
                    jkState eval_state;
                    new_state.copy_to( eval_state );
                    eval_state.resort_friends();
                    //eval_state.resort_opps();
                    PlayerSet team2 = WSinfo::alive_teammates_without_me;
                    team2.keep_players_in_circle( possible_pos[ j ], 3.0 );
                    //#ifdef DEBUG
                    //attention: overkill; too many circles
                    //DBLOG_DRAW(LOG_LEVEL,C2D(possible_pos[j].x,possible_pos[j].y,4.0, "red"));
                    //#endif
                    if( ( team2.num <= 2 ) && in_field( possible_pos[ j ] ) && in_range_pos( possible_pos[ j ] ) ) //only if I would be there alone in order not to interfere
                    {
#ifdef DEBUG
                        DBLOG_DRAW( LOG_LEVEL, VC2D (possible_pos[j], 0.1, "white") );
                        DBLOG_POL( LOG_LEVEL, "\njkState:\n" << eval_state.toString () );
#endif
                        temp_V = get_V_for_State( eval_state );
                        if( team2.num >= 1 ) //would like to be alone, but if a teammate is unavoidable go as far away as possible
                            temp_V += ( 3.0 - team2[ 0 ]->pos.distance( possible_pos[ j ] ) );
#ifdef DEBUG
                        DBLOG_POL( LOG_LEVEL, "position j: " << j << " ; value: " << temp_V << " ; (" << possible_pos[j].getX () << "/" << possible_pos[j].getY () << ")\n" );
#endif
                        posis->add( j, temp_V );
                        posi_count2++;
                    }
                }
            }
            else //do not use a model but assume all others are staying for simplicity
            {
                jkState new_state;
                new_state.get_from_WS( bholder );
                for( int j = 0; j < posi_count; j++ )
                {
                    new_state.set_friend_pos( my_friend_num, possible_pos[ j ] );
                    PlayerSet team2 = WSinfo::alive_teammates;
                    team2.keep_players_in_circle( possible_pos[ j ], 4.0 );
                    if( ( team2.num <= 1 ) && in_field( possible_pos[ j ] ) && in_range_pos( possible_pos[ j ] ) ) //only if I would be there alone in order not to interfere
                    {
                        team2.remove( WSinfo::me );
                        if( ( team2.num >= 1 ) && ( team2[ 0 ]->pos.distance( possible_pos[ j ] ) < 4.0 ) )
                            continue;
                        posis->add( j, get_V_for_State( new_state ) );
                        posi_count2++;
                    }
                }
            }
            int which;
            if( posi_count2 > 0 )
            {
                posis->do_sort(); //sort; smallest/cheapest first
#ifdef DEBUG
                DBLOG_POL( LOG_LEVEL, "best position's value: " << posis->get_value (0) );
#endif
                if( use_V_for_actions )
                {
                    if( ( ( use_exploration_nonballholder ) && ( drand48() < exploration ) ) && ( ( exploration_num < 0 ) || ( exploration_num == WSinfo::me->number ) ) )
                    { //select randomly
                        int temp = -1;
                        if( random_mode == 0 )
                        {
                            temp = floor( posi_count2 * ( drand48() - 0.000001 ) );
                            which = posis->get_key( temp );
#ifdef DEBUG
                            DBLOG_POL( LOG_LEVEL, "Nonballholder: selecting action randomly with uniform distribution - pos: (" << possible_pos[which].getX () << "/" << possible_pos[which].getY () << ") value: " << posis->get_value (temp) );
#endif
                        }
                        else if( random_mode == 1 )
                        {
                            double action_sum = 0.0;
                            double tmp_sum = 0.0;
                            for( temp = 0; temp < posi_count2; temp++ )
                                action_sum += ( 1 - posis->get_value( temp ) );
                            double target = drand48();
                            //every action gets a part of the cake (i.e. [0.0,1.0]) proportionally to (1-it's value) since high_value=high_costs=bad=>little probability
                            for( temp = 0; temp < posi_count2; temp++ )
                            {
                                tmp_sum += ( ( 1 - posis->get_value( temp ) ) / action_sum );
                                if( target < tmp_sum )
                                    break; //our arrow points to this part of the cake, so stop and take that action
                            }
                            which = posis->get_key( temp );
#ifdef DEBUG
                            DBLOG_POL( LOG_LEVEL, "Nonballholder: selecting action randomly with proportional distribution - pos: (" << possible_pos[which].getX () << "/" << possible_pos[which].getY () << ") value: " << posis->get_value (temp) );
#endif
                        }
                        else
                        {
#ifdef DEBUG
                            DBLOG_POL( LOG_LEVEL, "FATAL ERROR: Nonballholder: random_mode got undefined value" );
#endif
                        }
                    }
                    else
                    {
                        //which=posis->get_key(0);  //best value
                        //begin randomize between best positions if there are several with the same value
                        int ix = 0;
                        double bval = posis->get_value( 0 );
                        if( bval > REJECT_VAL ) //we have not a very good clue so better let default decide...
                            return false;
                        //cerr <<"BEST ACTIONS VALUE "<<bval<<"\n";
                        while( ( posi_count2 > ix + 1 ) && ( posis->get_value( ix + 1 ) == bval ) )
                            ix++;
                        ix = rand_in_range( 0, ix );
                        //end randomize between best positions if there are several with the same value
                        which = posis->get_key( ix );
                        //val=eval_actions->get_value(ix);
                    }
                }
                else
                    which = rand_in_range( 0, posi_count2 - 1 ); //random selection
            }
            else
            { //no position found - this should never happen...
                delete[] possible_pos;
                return false;
            }
            go2pos->set_target( possible_pos[ which ], .2, 0, 0 ); //jk was before: go2pos->set_target(possible_pos[which],.5,1,true);
            if( go2pos->get_cmd( cmd ) )
            {
                last_good_pos = possible_pos[ which ]; // not currently needed
#ifdef DEBUG
                DBLOG_POL( LOG_LEVEL, "selected position: " << which << "; (" << possible_pos[which].getX () << "/" << possible_pos[which].getY () << ")" );
                DBLOG_DRAW( LOG_LEVEL, VC2D (possible_pos[which], 0.3, "red") );
#endif
                cached_no_ball_pos = possible_pos[ which ];
                delete[] possible_pos;
                return true;
            }
            else
            {
#ifdef DEBUG
                DBLOG_POL( LOG_LEVEL, "ERROR: COULDN'T GET CMD" );
#endif
            }
            delete[] possible_pos;
            possible_pos = NULL;
        }
        return false;
    }

    Intention intention;
//-----------------------------------
//I have got the ball
//-----------------------------------
#ifdef DEBUG
    DBLOG_POL( LOG_LEVEL, "got Ball, in Score04" );
#endif
    holding_ball_for++;
    cached_no_ball_pos.setY( -1 );
    result = jkwball03->score->test_shoot2goal( intention );
    if( result ) //shoot to goal - we have been successfull (almost for sure)!
        memorize( 0 );
    //keep previous intention if not complete yet
    if( !result )
    {
        result = jkwball03->check_previous_intention( jkwball03->my_blackboard.intention, intention );
        if( result )
        { //OK, we are still in an option and thus continue
#ifdef DEBUG
            DBLOG_POL( LOG_LEVEL, "still in an option which is still valid, continue it" );
#endif
            memorize( 999 );
            new_intention = false;
        }
    }

    if( !result )
    {
        jkState state;
        state.get_from_WS( WSinfo::me );
        int max_actions = 30;
        int num_actions = 0;
        Score04Action *actions = new Score04Action[ max_actions ]; //architecture allows for filtering,
        //which later might be applied to vary the selectable actions
        //direct pass to teammate
        PlayerSet team = WSinfo::alive_teammates;
        team.remove( WSinfo::me );
        team.keep_and_sort_closest_players_to_point( team.num, WSinfo::me->pos );
        for( i = 0; i < number_of_attackers; i++ )
        {
            if( ! ( ( holding_ball_for <= 2 ) && ( last_teammate_excl_me_at_ball == team[ i ]->number ) ) )
            {

                //either allow immediate backpasses:
                actions[ num_actions++ ].set_pass( i + 1 );
            }
            //or do not allow immediate backpasses:
            /*      if((state.friend_pos[i].x<WSinfo::his_team_pos_of_offside_line()-1.0)&&(WSinfo::me->pos.distance(state.friend_pos[i])>4.0))
             actions[num_actions++].set_pass(i+1);
             }
             else
             {
             #ifdef  DEBUG
             DBLOG_POL(LOG_LEVEL,"Passing back to "<<last_teammate_excl_me_at_ball<<" not considered valid action\n");
             #endif
             //do not consider immediate passes back to the player I got it from. This is only for reasons of the pass technique:
             //such immediate backpasses almost always fail and thus destory our statistics
             }*/
        }
        /*  //(laufpass) pass to position in teammate action radius
         for(i=0;i<NUM_FRIENDS;i++){
         if(state.friend_pos[i].x<WSinfo::his_team_pos_of_offside_line()-2.0)
         continue;
         float dist_to=state.friend_pos[i].distance(WSinfo::me->pos);
         if(dist_to>15.0)
         continue;
         WSpset team=WSinfo::alive_teammates_without_me;
         team.keep_and_sort_closest_players_to_point(NUM_FRIENDS,WSinfo::me->pos);
         PPlayer pp=team[i];
         if(pp==NULL)
         continue;
         float action_range=(dist_to / 2.5)-1.0; //rough approximation
         int resolution=18;
         Vector possible_pos [resolution+2];
         for(int j=0;j<resolution;j++){
         possible_pos[j]=Vector(ANGLE(2*M_PI / resolution * j));
         possible_pos[j].normalize(1.0+(action_range*drand48()));
         possible_pos[j]+=state.friend_pos[i];
         actions[num_actions++].set_pass(i+1,possible_pos[j],pp->number);
         }
         }
         */
        //dribblestraight
        actions[ num_actions++ ].set_dribble();

        //immediate_selfpasses currently not used
        //for(i=0;i<8;i++)
        //  actions[num_actions++].set_immediate_selfpass(DEG2RAD((360.0/8*i)+1.0));

        //selfpasses:
        ANGLE kickdir;
        Vector target;
        double speed;
        int steps;
        Vector dummy1;
        int dummy2;
        double opp_time2react = 1.0;
        for( i = 0; i < 8; i++ )
        {
            kickdir = ANGLE( DEG2RAD( (360.0 / 8 * i) + .1 ) );
            target.init_polar( 1.0, kickdir );
            target += WSinfo::ball->pos;
            if( selfpass->is_selfpass_safe( kickdir, speed, target, steps, dummy1, dummy2, 2, opp_time2react ) )
            {
                actions[ num_actions ].param2 = speed;
                actions[ num_actions ].target = target;
                actions[ num_actions++ ].set_selfpass( kickdir.get_value(), speed, target );
            }
#ifdef DEBUG
//DBLOG_DRAW(LOG_LEVEL,C2D(target.x,target.y,0.1, "white"));
#endif
        }

        float bestval = -1.0;
        //evaluate actions and Sort by expected Value in Sort * eval_actions, returns array index of best action
        int which_action = select_action( actions, num_actions, state, bestval );

        //select the action with the expected highest payoff
        if( ( ( !use_V_for_actions ) || ( ( train_mode ) && ( drand48() < exploration ) ) ) && ( ( exploration_num < 0 ) || ( exploration_num == WSinfo::me->number ) ) )
        { //select a random action
            if( !use_V_for_actions )
            {
                float x = drand48();
                if( x <= 0.33 )
                    which_action = rand_in_range( 0, 10 );
                if( ( x > 0.33 ) && ( x <= 0.66 ) )
                    which_action = 2; //dribble
                if( x > 0.66 )
                    which_action = rand_in_range( 3, 10 );
#ifdef DEBUG
                DBLOG_POL( LOG_LEVEL, "absolutely randomly (i.e. without success chance check) selected action of type " << actions[which_action].type );
#endif
            }
            else
            { //eploration
                int temp = -1;
                if( random_mode == 0 )
                {
                    temp = floor( eval_action_cnt * ( drand48() - 0.000001 ) );
                    which_action = eval_actions->get_key( temp );
#ifdef DEBUG
                    DBLOG_POL( LOG_LEVEL, "selecting action randomly with uniform distribution - type: " << actions[which_action].type << " value: " << eval_actions->get_value (temp) );
#endif
                }
                else if( random_mode == 1 )
                {
                    double action_sum = 0.0;
                    double tmp_sum = 0.0;
                    for( temp = 0; temp < eval_action_cnt; temp++ )
                        action_sum += ( 1 - eval_actions->get_value( temp ) );
                    double target = drand48();
                    //every action gets a part of the cake (i.e. [0.0,1.0]) proportionally to (1-it's value) since high_value=high_costs=bad=>little probability
                    for( temp = 0; temp < eval_action_cnt; temp++ )
                    {
                        tmp_sum += ( ( 1 - eval_actions->get_value( temp ) ) / action_sum );
                        if( target < tmp_sum )
                            break; //our arrow points to this part of the cake, so stop and take that action
                    }
                    which_action = eval_actions->get_key( temp );
#ifdef DEBUG
                    DBLOG_POL( LOG_LEVEL, "selecting action randomly with proportional distribution - type: " << actions[which_action].type << " value: " << eval_actions->get_value (temp) );
#endif
                }
                else
                {
#ifdef DEBUG
                    DBLOG_POL( LOG_LEVEL, "FATAL ERROR: random_mode got undefined value" );
#endif
                }
            }
        }
        else // select action with the lowest expected costs
        {
            if( ( which_action == -1 ) || ( bestval > REJECT_VAL ) )
            {
                if( train_mode )
                {
                    which_action = rand_in_range( 0, num_actions );
#ifdef DEBUG
                    DBLOG_POL( LOG_LEVEL, "NO GOOD ACTION FOUND (HOW COMES?), DOING SOMETHING AT RANDOM" );
#endif
                }
                else
                    return false;
            }
        }
        if( which_action > -1 )
        {
            //cerr <<"BEST ACTIONS VALUE: " <<bestval<<" \n";
            Score04Action do_it = actions[ which_action ];

            if( train_mode )
                memorize( actions[ which_action ].type );

            //map the selected action to an intention
            result = apply_action( do_it, intention );
        }
        else
        {
#ifdef DEBUG
            cerr
                    << "SCORE04: No action found, let default wball03 decide; should not happen (fatal err 2).";
#endif
            delete[] actions;
            actions = NULL;
            return false;
        }
        delete[] actions;
        actions = NULL;
    }

    //always at the end:
    if( new_intention )
    {
        jkwball03->my_blackboard.pass_or_dribble_intention = intention;
        jkwball03->my_blackboard.intention = intention;
    }
    jkwball03->check_write2blackboard();
    jkwball03->intention2cmd( jkwball03->my_blackboard.intention, cmd );
    //cerr << "My INTENTIONc="<<jkwball03->my_blackboard.intention.get_type()<<"\n";
    if( jkwball03->my_blackboard.intention.valid_since() == 999999 )
    {
#ifdef DEBUG
        DBLOG_POL( LOG_LEVEL, "NO INTENTION" );
#endif
    }
#ifdef DEBUG
#if LOGGING && BASIC_LOGGING
    long endtime = Tools::get_current_ms_time();
#endif
    DBLOG_POL( LOG_LEVEL, "score04 took me " << (endtime - starttime) << "(ms)" );
#endif

    return result;
}

bool Score04::in_kickrange( Vector player_pos, Vector ball_pos )
{
    if( player_pos.distance( ball_pos ) <= ServerOptions::kickable_area )
        return true;
    else
        return false;
}

bool Score04::scan_field( Cmd &cmd )
{
    Tools::set_neck_request( NECK_REQ_SCANFORBALL );
    if( WSinfo::ws->view_quality == Cmd_View::VIEW_QUALITY_LOW )
    {
        cmd.cmd_body.set_turn( 0 );
        return true;
    }
    Angle turn = .5 * ( Tools::next_view_angle_width() + Tools::cur_view_angle_width() ).get_value();
    basiccmd->set_turn_inertia( turn );
    return basiccmd->get_cmd( cmd );
}
