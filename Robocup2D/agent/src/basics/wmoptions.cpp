#include "basics/wmoptions.h"

#include "stdlib.h"

#include "valueparser.h"
#include "macro_msg.h"

bool WMoptions::offline = false;

long WMoptions::ms_delta_max_wait_synch_see_long        = 40;
long WMoptions::ms_delta_max_wait_synch_see_short       = 30;
long WMoptions::ms_time_max_wait_after_sense_body_long  = 70;
long WMoptions::ms_time_max_wait_after_sense_body_short = 70;
long WMoptions::ms_time_max_wait_select_interval        = -1; //10;
long WMoptions::s_time_normal_select_interval           =  5;

bool WMoptions::foresee_opponents_positions             =  true;

bool WMoptions::use_fullstate_instead_of_see            =  true;
bool WMoptions::behave_after_fullstate                  =  true;
bool WMoptions::behave_after_think                      = false;
bool WMoptions::disconnect_if_idle                      =  true;
bool WMoptions::send_teamcomm                           = false;
bool WMoptions::recv_teamcomm                           = false;
bool WMoptions::ignore_fullstate                        = false;
bool WMoptions::ignore_sense_body                       = false;
bool WMoptions::ignore_see                              =  true;
bool WMoptions::ignore_hear                             = false;
bool WMoptions::use_pfilter                             =  true;
bool WMoptions::use_server_based_collision_detection    =  true;

int  WMoptions::max_cycles_to_forget_my_player          = 30;
int  WMoptions::max_cycles_to_forget_his_player         = 60;

bool WMoptions::save_msg_times                          = false;
int  WMoptions::his_goalie_number                       = 0;

/******************************************************************************/

void WMoptions::read_options( char const* file, int argc, char const* const * argv )
{
    ValueParser vp( file, "World_Model" );

    if( argc > 1 )
    {
        /* skip command name */
        argv++;
        argc--;

        vp.append_from_command_line( argc, argv, "wm_" );
    }
    bool dum;

    vp.get( "offline", offline );
    vp.get( "foresee_opponents_positions", foresee_opponents_positions );

    //vp.set_verbose(true);

    dum = false;
    vp.get( "test", dum );
    if( dum )
        set_mode_test();

    dum = !disconnect_if_idle;
    vp.get( "asynch", dum );
    disconnect_if_idle = !dum;

    vp.get( "use_pfilter", use_pfilter );
    vp.get( "use_server_based_collision_detection", use_server_based_collision_detection );
    dum = false;
    vp.get( "synch_mode", dum );
    if( dum )
        set_mode_synch_mode();

    dum = false;
    vp.get( "synch_mode_full", dum );
    if( dum )
        set_mode_synch_mode_with_fullstate();

    dum = send_teamcomm;
    vp.get( "send_teamcomm", dum );
    if( dum != send_teamcomm )
    {
        send_teamcomm = dum;
        cout << "\nwm_send_teamcomm set to: " << send_teamcomm;
    }
    dum = recv_teamcomm;
    vp.get( "recv_teamcomm", dum );
    if( dum != recv_teamcomm )
    {
        recv_teamcomm = dum;
        cout << "\nrecv_teamcomm set to: " << recv_teamcomm;
    }

    vp.get( "ms_wait", ms_time_max_wait_after_sense_body_long );
    vp.get( "ms_wait_short", ms_time_max_wait_after_sense_body_short );
    vp.get( "s_wait", s_time_normal_select_interval );
    if( vp.get( "mta", max_cycles_to_forget_my_player ) > 0 )
        cout << "\nmax_cycles_to_forget_my_player= "
                << max_cycles_to_forget_my_player << flush;

    if( vp.get( "hta", max_cycles_to_forget_his_player ) > 0 )
        cout << "\nmax_cycles_to_forget_his_player= "
                << max_cycles_to_forget_his_player << flush;

    vp.get( "save", save_msg_times );

    vp.get( "his_goalie_number", his_goalie_number );

    vp.get( "ms_wait_si", ms_time_max_wait_select_interval );

    if( vp.num_of_not_accessed_entries() )
    {
        ERROR_OUT
                << "\nInput: not recognized world model options (prefix wm_*):";
        vp.show_not_accessed_entries( ERROR_STREAM );
        ERROR_STREAM << "\nexiting ...\n";
        exit( 1 ); //for the moment
    }
}

void WMoptions::set_mode_competition()
{
    use_fullstate_instead_of_see = false;
    send_teamcomm = true;
    recv_teamcomm = true;
    disconnect_if_idle = true;
    behave_after_fullstate = false;
    behave_after_think = false;
    ignore_fullstate = true;
    ignore_sense_body = false;
    ignore_see = false;
    ignore_hear = false;
}

void WMoptions::set_mode_test()
{
    use_fullstate_instead_of_see = false;
    send_teamcomm = true;
    recv_teamcomm = true;
    disconnect_if_idle = true;
    behave_after_fullstate = false;
    behave_after_think = false;
    ignore_fullstate = false;
    ignore_sense_body = false;
    ignore_see = false;
    ignore_hear = false;
}

void WMoptions::set_mode_synch_mode()
{
    use_fullstate_instead_of_see = false;
    send_teamcomm = true;
    recv_teamcomm = true;
    disconnect_if_idle = true;
    behave_after_fullstate = false;
    behave_after_think = true;
    ignore_fullstate = false;
    ignore_sense_body = false;
    ignore_see = false;
    ignore_hear = false;
}

void WMoptions::set_mode_synch_mode_with_fullstate()
{
    use_fullstate_instead_of_see = true;
    send_teamcomm = true;
    recv_teamcomm = true;
    disconnect_if_idle = true;
    behave_after_fullstate = false;
    behave_after_think = true;
    ignore_fullstate = false;
    ignore_sense_body = false;
    ignore_see = false;
    ignore_hear = false;
}
