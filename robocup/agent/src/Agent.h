#ifndef AGENT_SRC_AGENT_H_
#define AGENT_SRC_AGENT_H_

#include "Communicator.h"

#include "basics/mdp_memory.h"




#include "behaviors/base_bm.h"

// BodyBehavior
#include "bs03_bmc.h"
#include "train_behaviors_bmc.h"
#include "train_skills_bmc.h"
#include "test_skills_bmc.h"
#include "artplayer_bmc.h"  /* Artur's test player */
#include "sputplayer_bmc.h" /* Sputnick's test player... */
#include "infprojrss_bmc.h" /* test player for the Guided Tutorial of the Robotic Soccer Simulation Project */

// NeckBehavior
#include "view/bs03_neck_bmn.h"  /* turn_neck strategy #42 */
//#include "view/goalie_neck_bmn.h" /* goalie_dynamic_neck */

// ViewBehavior
#include "view/synch_view08_bmv.h" // bs03_view is obsolte due to new synchronous viewing
#include "view/bs03_view_bmv.h"  /* view strategy #41 */

// AttentionToBehavior
#include "view/attention_to_bma.h"

// PointToBehavior
#include "pointto10_bmp.h"

#include "view/neck_and_view17.h"



class Agent
{

private:

    Communicator comm;
    MessagesInfo msg_info;
    CommandsInfo cmd_info;

    bool server_alive;

    WMstate  wm,     wmfull;
    WS       wstate, wstate_full;
    MDPstate mdp,    mdpfull;

    MDPmemory *mdp_memory;

    Cmd cmd;
    Cmd last_cmd;

    BodyBehavior        *body_controller;
    NeckBehavior        *neck_controller;
    ViewBehavior        *view_controller;
    NeckViewBehavior	*neck_view_controller;
    AttentionToBehavior *attentionto_controller;
    PointToBehavior     *pointto_controller;

public:

    Agent( int argc, char *argv[] );
    virtual ~Agent();

    Agent& run();

private:

    void check_reduce_dash_power( Cmd &cmd_form );

    void export_mdpstate( MDPstate &mdp_state, MDPstate &mdp_state_full, const WMstate *state, const WMstate *state_full);
    void export_ws(       WS       *ws_state,  WS       *ws_state_full,  const WMstate *state, const WMstate *state_full);

    bool init_body_behavior(        BodyBehavior        *&behavior, int argc, char const* const* argv );
    bool init_neck_behavior(        NeckBehavior        *&behavior, int argc, char const* const* argv );
    bool init_view_behavior(        ViewBehavior        *&behavior, int argc, char const* const* argv );
    bool init_neckview_behavior(    NeckViewBehavior    *&behavior, int argc, char const* const* argv );
    bool init_attentionto_behavior( AttentionToBehavior *&behavior, int argc, char const* const* argv );
    bool init_pointto_behavior(     PointToBehavior     *&behavior, int argc, char const* const* argv );
};

#endif /* AGENT_SRC_AGENT_H_ */
