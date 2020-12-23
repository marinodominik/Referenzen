#include "Agent.h"

#include "log_macros.h"

#include "mdp_info.h"

#include <stdlib.h>
#include <iomanip>

#include "messages_times.h"
#include "serverparam.h"
#include "basics/ws_memory.h"
#include "basics/blackboard.h"
#include "policy/policy_tools.h"
#include "policy/planning.h"




#include <ostream>

using namespace std;

ofstream gvDummyOutputStream("/dev/null");



Agent::Agent( int argc, char *argv[] )
{
    msg_info.set_cycle( -1 );
    msg_info.set_ms_time( 0 );

    cmd_info.reset();

    server_alive = false;

    mdp_memory = 0;

    body_controller = 0;
    neck_controller = 0;
    view_controller = 0;
    neck_view_controller = 0;
    attentionto_controller = 0;
    pointto_controller = 0;

    mdpInfo::my_current_cmd = &cmd;
    WSinfo::current_cmd     = &cmd;
    WSinfo::last_cmd        = &last_cmd;

    //cout << endl << "[CommandLineOptions";
    CommandLineOptions::init_options();
    CommandLineOptions::read_from_command_line(argc,argv);
    //cout << "|WMoptions";
    WMoptions::read_options(CommandLineOptions::agent_conf,argc,argv);

    //init default values for agent
    //cout << "|ClientOptions";
    ClientOptions::init();
    //read agent's default parameter file
    ClientOptions::read_from_file( CommandLineOptions::agent_conf );
    ClientOptions::read_from_command_line(argc,argv);
    strncpy(WM::my_team_name,ClientOptions::teamname,20);
    WM::my_team_name_len= strlen(WM::my_team_name);

    MessagesTimes::init(WMoptions::save_msg_times);

    if( ClientOptions::server_version < 8.00 )
    {
        cout << "Error support for server_version < 8.00 was dropped" << endl;
        exit( EXIT_SUCCESS );
    }

    if( !( comm.initialize() ) )
    {
        ERROR_OUT << ID << "\nProblems with initialization connection to " << CommandLineOptions::host << " at port " << CommandLineOptions::port;
        exit( EXIT_FAILURE );
    }

    comm.send_initialize_message();

    PlayMode pm;
    if( !comm.recv_initialize_message( pm ) )
    {
        ERROR_OUT << ID << "\nProblems with initialization connection  to " << CommandLineOptions::host;
        exit( EXIT_FAILURE );
    }

    server_alive = true;

    ClientOptions::player_no = WM::my_number;

    if (WMoptions::his_goalie_number > 0 && WMoptions::his_goalie_number <= 11)
    {
        wm.his_goalie_number     = WMoptions::his_goalie_number;
        wmfull.his_goalie_number = WMoptions::his_goalie_number;
    }


    // normal way is to simply parse the parameter messages according
    // to the known rules

    // >>>>>>>>>>>> uncomment for standard use
    if( !comm.recv_parameter_messages() )
    {
        ERROR_OUT << ID << "\nProblems with initialization of server parameters";
        exit( EXIT_FAILURE );
    }
    // <<<<<<<<<<<< stop uncomment for standard use

    // use this option instead to create a new default_server_param.h
    // that is written to the speciefied output stream
    // after this is done you can use i.e.
    // produce_server_param_parser("server_param")
    // to create a parser for the server parameters

    // >>>>>>>>>>>> uncomment first
//    ofstream param_file;
//    param_file.open("default_server_param.h");
//
//    cout << "\ndone producing parameter messages, result code is res= "
//            << comm.produce_parameter_messages(param_file) << "\n";
//
//    param_file.close();
//    exit(0);
    // <<<<<<<<<<<< stop uncomment first

    // make sure you have the right default_server_param.h in place
    // before using this and please note that new parameters have to be
    // put into sensorbuffer.h and server_options.h
    // >>>>>>>>>>>> uncomment 2
//    produce_server_param_parser("server_param");
//    produce_server_param_parser("player_param");
//    exit(0);
    // <<<<<<<<<<<< stop uncomment 2

    if( !ServerParam::export_server_options() )
    {
        ERROR_OUT << ID << "\nProblems with exporting server options";
        exit( EXIT_FAILURE );
    }

    //cout << "\n[ServerOptions] ok!";
    //cout << "\nServerOptions::player_size= " << ServerOptions::player_size;

    wmfull.init();
    wm.init(); //call this afer the WM::my_number is known!
    wm.play_mode = pm;

    if(ClientOptions::view_type == SYNCH_SEE)
    {
        // start synch view mode
        comm.send_string_directly( "(synch_see)" );
    }

    LogOptions::init();
    LogOptions::read_from_file(CommandLineOptions::agent_conf);
    LogOptions::read_from_command_line(argc,argv);

    sprintf(LogOptions::log_fname,"%s/%s%d-%c-actions.log",LogOptions::log_dir, ClientOptions::teamname, ClientOptions::player_no, left_SIDE == WM::my_side ? 'l' : 'r' );
    LogOptions::open();

    //  cout << endl << "[mdpInfo";
    mdpInfo::init();
    //  cout << "|WSmemory";
    WSmemory::init();
    //  cout << "|Blackboard";
    Blackboard::init();
    //  cout << "|Policy_Tools";
    Policy_Tools::init_params();
    //  cout << "|Planning";
    Planning::init_params();
    //  cout << "|AbstractMDP]";
    AbstractMDP::init_params();
    //  cout << "|DeltaPositioning]";
    DeltaPositioning::init_formations();
    //  cout << " ok!" << endl;

    /* Sput: We still need MDPmemory, and we need it in client.cpp... */
    mdp_memory = new MDPmemory();


    if( !init_body_behavior( body_controller, argc, argv ) )
        exit( EXIT_FAILURE );

    if( ClientOptions::use_neckview )
    {
		if( !init_neckview_behavior( neck_view_controller, argc, argv ) )
			exit( EXIT_FAILURE );
    }
    else
    {
		if( !init_neck_behavior( neck_controller, argc, argv ) )
			exit( EXIT_FAILURE );

		if( !init_view_behavior( view_controller, argc, argv ) )
			exit( EXIT_FAILURE );
    }

    if( !init_attentionto_behavior( attentionto_controller, argc, argv ) )
        exit( EXIT_FAILURE );

    if( !init_pointto_behavior( pointto_controller, argc, argv ) )
        exit( EXIT_FAILURE );


    if( WMoptions::offline )
    {
        if( !body_controller )
        {
            ERROR_OUT << ID << "\nneed a behavior in offline mode";
            exit( EXIT_FAILURE );
        }

        WMstate wm;
        WS wstate;
        export_ws( &wstate, 0, &wm, 0 );

        Cmd cmd;
        bool res = body_controller->get_cmd( cmd );

        INFO_OUT << ID << "\nresult of offline mode execution " << res << "\n";
        exit( EXIT_SUCCESS );
    }


    comm.set_socket_nonblocking();

    comm.send_string_directly( "(clang (ver 7 8))" ); //this is important to get coach clang messages (from server ver. 8.04 rel 5)

    if( WM::my_number == 1 )
    {
        CommandLineOptions::print_options();
        cout << endl << flush;
    }

    cout << "Agent No. " << setw(2) << WM::my_number << " is started and ready to rumble!" << endl << flush;
}

Agent::~Agent()
{
    comm.send_string_directly( "(bye)" );
    //cout << "\nShutting down..." << endl; // player " << WM::my_number << endl;
    mdpInfo::print_mdp_statistics();

    if( WMoptions::save_msg_times )
    {
        char buf[ 256 ];
        sprintf( buf, "%s2", LogOptions::log_fname );
        MessagesTimes::save( buf );
    }

    /*
  if( LogOptions::log_int ) {
    IntentionLogger::save_all_until_now();
  }
     */

    LogOptions::close();

    //cout << "\nShutting down player " << WM::my_number << "\n" << endl << flush;
}

Agent& Agent::run()
{
    while( server_alive )
    {
        msg_info.reset();
        mdp.reset();
        mdpfull.reset();
        cmd.reset();

        int retval = comm.idle( 0, WMoptions::s_time_normal_select_interval * ServerOptions::slow_down_factor);

        if( retval == 0 && WMoptions::disconnect_if_idle )
        {
            server_alive = false;
            continue;
        }

        if( comm.has_data_on_socket() )
        {
            comm.receive_and_incorporate_server_messages( msg_info, cmd_info, &wm, &wmfull, body_controller, last_cmd );
        }

        if( !msg_info.msg[ MESSAGE_SENSE_BODY ].processed &&
            !msg_info.msg[ MESSAGE_FULLSTATE  ].processed &&
            !msg_info.msg[ MESSAGE_THINK      ].processed )
        {
            continue;
        }

        if ( WMoptions::behave_after_think )
        {
            if( !msg_info.msg[ MESSAGE_THINK ].processed )
            {
                continue;
            }

            bool have_fullstate = ( msg_info.msg[ MESSAGE_FULLSTATE ].cycle == msg_info.msg[ MESSAGE_THINK ].cycle );

            if( WMoptions::use_fullstate_instead_of_see )
            {
                if( !have_fullstate )
                    WARNING_OUT << ID << "no fullstate information for this cycle, using old info";
                export_mdpstate( mdp, mdpfull,      0, &wmfull );
                export_ws(       0,   &wstate_full, 0, &wmfull );
            }
            else if( have_fullstate )
            { //just using fullstate for comparisson!
                export_mdpstate( mdp,     mdpfull,      &wm, &wmfull );
                export_ws(       &wstate, &wstate_full, &wm, &wmfull );
            }
            else
            { //this is the most usual case
                export_mdpstate( mdp,     mdpfull, &wm, 0 );
                export_ws(       &wstate, 0,       &wm, 0 );
            }
        }
        else if( WMoptions::behave_after_fullstate )
        {
            if( !msg_info.msg[ MESSAGE_FULLSTATE ].processed )
                continue;
            if( !WMoptions::use_fullstate_instead_of_see )
                WARNING_OUT << ID << "\nuse_fullstate_instead_of_see==false doesn't make sense here";
            export_mdpstate( mdp, mdpfull,      0, &wmfull );
            export_ws(       0,   &wstate_full, 0, &wmfull );
        }
        else if( msg_info.msg[ MESSAGE_SENSE_BODY ].processed )
        {
            while (true)
            {
                double my_distance_to_ball = wm.my_distance_to_ball();
                long ms_time = Tools::get_current_ms_time();
                long ms_time_we_can_still_wait = ( msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time - ms_time )
                                                 * ServerOptions::slow_down_factor;

                if( wm.synch_mode_ok || Blackboard::get_guess_synched() )
                {
                    if( my_distance_to_ball < 5.0 )
                        ms_time_we_can_still_wait += ServerOptions::synch_see_offset
                        + WMoptions::ms_delta_max_wait_synch_see_long;
                    else
                        ms_time_we_can_still_wait += ServerOptions::synch_see_offset
                        + WMoptions::ms_delta_max_wait_synch_see_short;
                }
                else
                {
                    if ( my_distance_to_ball < 5.0 )
                        ms_time_we_can_still_wait += WMoptions::ms_time_max_wait_after_sense_body_long
                        * ServerOptions::slow_down_factor;
                    else
                        ms_time_we_can_still_wait += WMoptions::ms_time_max_wait_after_sense_body_short
                        * ServerOptions::slow_down_factor;
                }

                long ms_time_till_next_see = msg_info.msg[ MESSAGE_SEE ].ms_time
                        + (wm.ms_time_between_see_messages()
                                * ServerOptions::slow_down_factor )
                                - ms_time;

                if( ms_time_we_can_still_wait <= 0 )
                    break;

                bool wait = msg_info.msg[ MESSAGE_FULLSTATE ].cycle >= 0 && //the fullstate messages are activated
                            msg_info.msg[ MESSAGE_FULLSTATE ].cycle <  msg_info.msg[ MESSAGE_SENSE_BODY ].cycle;

                if( !wait )
                {
                    if( wm.synch_mode_ok || Blackboard::get_guess_synched() )
                    { // in synch viewing mode
                        wait = msg_info.msg[ MESSAGE_SEE ].ms_time < msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time;
                    }
                    else
                    { // not in synch viewing mode
#if 1 //original code, everybody waits
                        wait = true
#else
                                wait = ( my_distance_to_ball() <  15.0
                                        || ClientOptions::consider_goalie && wm.my_distance_to_ball() < 35.0 )
#endif
                                        && ( ms_time_till_next_see < (ms_time_we_can_still_wait + ( 15 * ServerOptions::slow_down_factor ) ) )
                                        // + X * slowdownfactor as tolerance, if messages come earlier then expected
                                        //&& msg_info.msg[MESSAGE_SEE].cycle < msg_info.msg[MESSAGE_SENSE_BODY].cycle;
                                        && msg_info.msg[ MESSAGE_SEE ].ms_time < msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time;
                    }
                    if( !wait )
                        break;
                }

                long ms_time_dum = ms_time_we_can_still_wait;

                if( ( WMoptions::ms_time_max_wait_select_interval * ServerOptions::slow_down_factor ) > 0
                        && ms_time_dum > ( WMoptions::ms_time_max_wait_select_interval * ServerOptions::slow_down_factor ) ) //just make the time slots smaller (because of system load)
                    ms_time_dum = WMoptions::ms_time_max_wait_select_interval * ServerOptions::slow_down_factor;

                retval = comm.idle( ms_time_dum, 0 );
                //retval = idle( ms_time_we_can_still_wait, 0, sock.socket_fd );

                msg_info.reset();

                if( comm.has_data_on_socket() )
                    comm.receive_and_incorporate_server_messages( msg_info, cmd_info, &wm, &wmfull, body_controller, last_cmd );

                if( msg_info.msg[ MESSAGE_SENSE_BODY ].processed )
                {
                    //never use LOG_ERR here, because mdp could be not initialized
                    LOG_WM_ERR( msg_info.msg[ MESSAGE_SENSE_BODY ].received, 0, << "[" << WM::my_number << "] ERROR, got 2 SENSE BODY WITHOUT BEHAVE");
                    ERROR_OUT << ID << "\nERROR, got 2 SENSE BODY WITHOUT BEHAVE" << flush;
                    break;
                }

            } //end of while loop


            //exporting the mdpstate
            if( msg_info.msg[ MESSAGE_FULLSTATE ].cycle == msg_info.msg[ MESSAGE_SENSE_BODY ].cycle )
            {
                if( wm.my_distance_to_ball() <= 3.0 )
                {
                    if(msg_info.msg[ MESSAGE_SEE ].cycle == msg_info.msg[ MESSAGE_SENSE_BODY ].cycle )
                    {
                        LOG_WM( wm.time.time, 0, << "GOT_SEE 1" );
                    }
                    else
                    {
                        LOG_WM( wm.time.time, 0, << "GOT_SEE 0" );
                    }
                }

                wm.compare_with_wmstate( wmfull );

#if 1   //this is the correct setting!!!
                export_mdpstate( mdp,     mdpfull,      &wm, &wmfull );
                export_ws(       &wstate, &wstate_full, &wm, &wmfull );
#else   //just DEBUGGING, never leave it so to the repository
                if ( true || msg_info.msg[MESSAGE_SEE].cycle == msg_info.msg[MESSAGE_SENSE_BODY].cycle )
                {
                    export_mdpstate(mdp,mdpfull,&wmfull,&wmfull);
                    export_ws(&wstate,&wstate_full,&wmfull,&wmfull);
                }
                else
                {
                    export_mdpstate(mdp,mdpfull,&wm,&wmfull);
                    export_ws(&wstate,&wstate_full,&wm,&wmfull);
                }
#endif
            }
            else
            {
                export_mdpstate( mdp,     mdpfull, &wm, 0 );
                export_ws(       &wstate, 0,       &wm, 0 );
            }
        }
        else
            continue;



        //the rest of the loop performs a behave of the agent
        wm.show_object_info();
        long ms_time = Tools::get_current_ms_time(); //actualize current time
        MessagesTimes::add_before_behave( WSinfo::ws->time, ms_time );

        //some extra time information for the behave (a hack, should happen in the export methods)
        mdp.ms_time_of_sb         = msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time;
        mdpfull.ms_time_of_sb     = msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time;
        wstate.ms_time_of_sb      = msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time;
        wstate_full.ms_time_of_sb = msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time;

        WSmemory::update();

        /* we still need MDPmemory... */
        mdp_memory->update();

        Tools::setModelledPlayer(WSinfo::me); // re-initialization



        /* Sput03: we need to call view strategy first!
       Because the old view strategy is too deeply nested with Bs2kAgent, it is better
       to convert it into a behavior structure. Since the later-to-be-used mechanism for view
       and neck behaviors has not yet been determined, this may only be a hack for now. */
        /* Some old policies reset the cmd, so we need to preserve its state...
       We should be able to remove this hack after switching completely to behaviors. */
		Cmd view_cmd;
        if( !ClientOptions::use_neckview )
        {
			if( view_controller )
				view_controller->get_cmd( view_cmd );
        }

        if ( body_controller)
            body_controller->get_cmd(cmd);

        check_reduce_dash_power(cmd);

        if( !ClientOptions::use_neckview )
        {
			/* see above */
			if( view_cmd.cmd_view.is_cmd_set() )
			{
				int view_ang,view_qual;
				view_cmd.cmd_view.get_angle_and_quality(view_ang,view_qual);
				cmd.cmd_view.set_angle_and_quality(view_ang,view_qual);
			}
        }

        if( pointto_controller ) /* JTS10: first call pointto controller */
        {
            pointto_controller->get_cmd(cmd);
        }

        if( !ClientOptions::use_neckview )
        {
			if( neck_controller ) /* Sput03: Now we need to call our neck behavior. */
			{
				neck_controller->get_cmd(cmd);
			}
        }

        if( ClientOptions::use_neckview )
        {
        	neck_view_controller->get_cmd(cmd);
        }

        if( attentionto_controller )
        {
            attentionto_controller->get_cmd(cmd);
        }

        ms_time = Tools::get_current_ms_time(); //actualize current time

        MessagesTimes::add_after_behave( WSinfo::ws->time, ms_time );

        last_cmd = cmd;

        long ms_time_dum = Tools::get_current_ms_time();

#ifdef CMD_VIEW_HACK
        if( !WMoptions::behave_after_think ) //do not use view mode synchronization in synch_mode
            if( cmd.cmd_view.is_cmd_set()
                    //          && msg_info.msg[MESSAGE_SEE].cycle != msg_info.msg[MESSAGE_SENSE_BODY].cycle ) {
                    && msg_info.msg[ MESSAGE_SEE ].ms_time < msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time )
            {
                comm.set_cmd_view_hack( cmd.cmd_view );
                cmd.cmd_view.unset_lock();
                cmd.cmd_view.unset_cmd();
            }
#endif

        /* Dieser Funktionsaufruf musste (in 2006) vor send_cmd gesetzt werden;
       anderenfalls hatte er keinen Einfluss. */
        wm.reduce_dash_power_if_possible( cmd.cmd_body ); //TG06

        //////send_cmd( sock, cmd, ms_time_left, msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time + 95);
        comm.send_cmd( cmd, wm.export_msg_teamcomm( cmd.cmd_say ) );//, tc, 0, msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time + 95); //never wait with say messages

        cmd_info.set_command( WSinfo::ws->time, cmd );
        wm.incorporate_cmd( cmd );
        WSmemory::incorporateCurrentCmd( cmd );

        //send_cmd( sock, cmd, ms_time_left ); //execute command

        if( ms_time_dum - ms_time > ( 40 * ServerOptions::slow_down_factor ) )
        {
            LOG_ERR( 0, << " Player " << WM::my_number << ": ms_time in behave = " << ms_time_dum - ms_time );
        }

        if( !WMoptions::ignore_sense_body
                && ms_time_dum - msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time > ( 80 * ServerOptions::slow_down_factor ) )
        {
            LOG_ERR( 0, << " Player " << WM::my_number << ": ms_time since sense_body " << ms_time_dum - msg_info.msg[ MESSAGE_SENSE_BODY ].ms_time );
        }

        if( WMoptions::behave_after_fullstate
                && ms_time_dum - msg_info.msg[ MESSAGE_FULLSTATE ].ms_time > ( 80 * ServerOptions::slow_down_factor ) )
        {
            LOG_ERR( 0, << " Player " << WM::my_number << ": ms_time since fullstate " << ms_time_dum - msg_info.msg[ MESSAGE_FULLSTATE ].ms_time );
        }
    }

    return *this;
}

void Agent::check_reduce_dash_power(Cmd &cmd_form)
{
    LOG_POL(0,<<"client.cpp: willMyStaminaCapacitySufficeForThisHalftime="
            <<Tools::willMyStaminaCapacitySufficeForThisHalftime());
    //modify dash-command according to available stamina
    if (cmd_form.cmd_body.get_type() == cmd_form.cmd_body.TYPE_DASH)
    {
        double power = 0;
        cmd_form.cmd_body.get_dash(power);
        LOG_POL(0, << "client.cpp:check_reduce_dash_power: Cmd Dash with power " << power);
        //LOG_DEB(0, << " Actual stamina " << mdpInfo::mdp->me->stamina.v);
        //LOG_DEB(0, << "stamina4steps_if_dashing " << mdpInfo::stamina4steps_if_dashing(power));
        //LOG_DEB(0, << "stamina_left " << mdpInfo::stamina_left());
        /*
    if (mdpInfo::mdp->me->effort.v < 0.99) {
      LOG_ERR(0, << "PROBLEM with stamina starts! Effort = " << mdpInfo::mdp->me->effort.v);
      }*/

        //if ((mdpInfo::stamina4steps_if_dashing(power) <= 0) || //VERY EXPENSIVE

        if ((ClientOptions::consider_goalie) && (cmd_form.cmd_body.get_priority() == 1))
        {
            LOG_ERR(0, << "Goalie dashes below stamina limit!");
            LOG_POL(0, << "Goalie dashes below stamina limit!");
        }
        else
        {
            LOG_POL(0,<<"client.cpp: stamina="<<WSinfo::me->stamina
                    <<" stamina_capacity="<<WSinfo::me->stamina_capacity);
            if ( WSinfo::me->stamina_capacity > 0 )//also use up anything, that's left
                if (((WSinfo::me->stamina <= ServerOptions::recover_dec_thr * ServerOptions::stamina_max + 12.0 - 2.0 * power) &&
                        (power < 0.0)) ||
                        ((WSinfo::me->stamina <= ServerOptions::recover_dec_thr * ServerOptions::stamina_max + 12.0 + power) &&
                                (power >= 0.0)))
                {
                    if (power < 0.0)
                    {
                        power = -0.5 * mdpInfo::stamina_left();
                    }
                    else
                    {
                        power = mdpInfo::stamina_left();
                    }
                    LOG_ERR(0, << "Warning (client.cpp): No stamina left: "
                            << WSinfo::me->stamina <<"  Reducing dash to " << power);
                    LOG_POL(0,<<"Warning (client.cpp): No stamina left: "
                            << WSinfo::me->stamina << " Reducing dash to " << power);

                    cmd_form.cmd_body.unset_lock();
                    cmd_form.cmd_body.set_dash(power);
                }
        }
    }
}

void Agent::export_mdpstate( MDPstate & mdp_state, MDPstate & mdp_state_full, const WMstate * state, const WMstate * state_full)
{
    if (state)
    {
        if (state_full)
        {
            state_full->export_mdpstate(mdp_state_full);
            mdpInfo::server_state= &mdp_state_full;
        }
        else
            mdpInfo::server_state= 0;

        state->export_mdpstate(mdp_state);
        mdpInfo::update(mdp_state);
        return;
    }

    if (state_full)
    {
        state_full->export_mdpstate(mdp_state_full);
        mdpInfo::server_state= &mdp_state_full;
        mdpInfo::update(mdp_state_full);
        return;
    }

    ERROR_OUT << ID << "\nshould never reach this point";
}

void Agent::export_ws( WS * ws_state, WS * ws_state_full, const WMstate * state, const WMstate * state_full)
{
    if (ws_state)
    {
        if (ws_state_full)
            state_full->export_ws(*ws_state_full);

        state->export_ws(*ws_state);

        WSinfo::init(ws_state,ws_state_full);
        return;
    }

    if (ws_state_full)
    {
        state_full->export_ws(*ws_state_full);
        WSinfo::init(ws_state_full,ws_state_full);
        return;
    }

    ERROR_OUT << ID << "\nshould never reach this point";
}

bool Agent::init_body_behavior( BodyBehavior *&behavior, int argc, char const* const* argv )
{
    //INFO_OUT << "\nlooking for behavior= [" << name << "]";

    behavior = 0;

    if( strcmp( ClientOptions::body_behavior, "Bs03" ) == 0 )
    {
        if( !Bs03::init( CommandLineOptions::agent_conf, argc, argv ) )
            return false;

        behavior = new Bs03();

        return true;
    }

    if( strcmp( ClientOptions::body_behavior, "TrainBehaviors" ) == 0 )
    {
        if( !TrainBehaviors::init( CommandLineOptions::agent_conf, argc, argv ) )
            return false;

        behavior = new TrainBehaviors();

        return true;
    }

    if( strcmp( ClientOptions::body_behavior, "TrainSkills" ) == 0 )
    {
        if( !TrainSkillsPlayer::init( CommandLineOptions::agent_conf, argc, argv ) )
            return false;

        behavior = new TrainSkillsPlayer();

        return true;
    }

    if( strcmp( ClientOptions::body_behavior, "TestSkillsPlayer" ) == 0 )
    {
        if( !TestSkillsPlayer::init( CommandLineOptions::agent_conf, argc, argv ) )
            return false;

        behavior = new TestSkillsPlayer();

        return true;
    }

    if( strcmp( ClientOptions::body_behavior, "ArtPlayer" ) == 0 )
    {
        if( !ArtPlayer::init( CommandLineOptions::agent_conf, argc, argv ) )
            return false;

        behavior = new ArtPlayer();

        return true;
    }

    if( strcmp( ClientOptions::body_behavior, "SputPlayer" ) == 0 )
    {
        if( !SputPlayer::init( CommandLineOptions::agent_conf, argc, argv ) )
            return false;

        behavior = new SputPlayer();

        return true;
    }

    if( strcmp( ClientOptions::body_behavior, "InfProjRSS" ) == 0 )
    {
        if( !InfProjRSS::init( CommandLineOptions::agent_conf, argc, argv ) )
            return false;

        behavior = new InfProjRSS();

        return true;
    }

    ERROR_OUT << "\ncould not find behavior= [" << ClientOptions::body_behavior << "]";

    return false;
}

bool Agent::init_neck_behavior( NeckBehavior *&behavior, int argc, char const* const* argv )
{
    if(!BS03Neck::init(CommandLineOptions::agent_conf,argc,argv))
        return false;

    behavior = new BS03Neck();

    return true;
}

bool Agent::init_view_behavior( ViewBehavior *&behavior, int argc, char const* const* argv )
{
    if( ClientOptions::view_type == SYNCH_SEE ) // we want the synchronous view Controller
    {
        if( !SynchView08::init( CommandLineOptions::agent_conf, argc, argv ) )
            return false;

        behavior = new SynchView08();

        return true;
    }
    else
    {
        if( !BS03View::init( CommandLineOptions::agent_conf, argc, argv ) )
            return false;

        behavior = new BS03View();

        return true;
    }
}

bool Agent::init_neckview_behavior( NeckViewBehavior *&behavior, int argc, char const* const* argv )
{
	if ( !NeckAndView17::init( CommandLineOptions::agent_conf, argc, argv ) )
	{
		return false;
	}

	behavior = new NeckAndView17();

	return true;
}

bool Agent::init_attentionto_behavior( AttentionToBehavior *&behavior, int argc, char const* const* argv )
{
    if( !AttentionTo::init( CommandLineOptions::agent_conf, argc, argv ) )
        return false;

    behavior = new AttentionTo();

    return true;
}

bool Agent::init_pointto_behavior( PointToBehavior *&behavior, int argc, char const* const* argv )
{
    if( !Pointto10::init( CommandLineOptions::agent_conf, argc, argv ) )
        return false;

    behavior = new Pointto10();

    return true;
}
