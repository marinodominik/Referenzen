#include "basics/WM.h"

#include "sensorbuffer.h"

#include "basics/server_options.h"

#include "../../lib/src/macro_msg.h"

char   WM::my_team_name[20];
int    WM::my_team_name_len;

int    WM::my_number;

int    WM::my_side;

double WM::my_radius;
double WM::my_kick_radius;
double WM::my_kick_margin;

double WM::my_inertia_moment;

double WM::my_decay;

double WM::my_dash_power_rate;

double WM::last_export_KONV;

int    WM::time;

void WM::set_my_type( Msg_player_type const *pt )
{
    if( !pt )
    {
        ERROR_OUT << ID << "zero type";
        return;
    }

    my_radius          = pt->player_size;
    my_kick_radius     = pt->player_size + pt->kickable_margin + ServerOptions::ball_size;
    my_kick_margin     = pt->kickable_margin;
    my_inertia_moment  = pt->inertia_moment;
    my_decay           = pt->player_decay;
    my_dash_power_rate = pt->dash_power_rate;
}

