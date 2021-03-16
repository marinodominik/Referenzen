#ifndef _WM_H_
#define _WM_H_

struct Msg_player_type;

class WM
{
public:
    static char   my_team_name[];
    static int    my_team_name_len;

    static int    my_number;

    static int    my_side;

    static double my_radius;
    static double my_kick_radius;
    static double my_kick_margin;

    static double my_inertia_moment;

    static double my_decay;

    static double my_dash_power_rate;

    static double last_export_KONV;

    static int    time;

    static void set_my_type( Msg_player_type const *pt );
};

#endif
