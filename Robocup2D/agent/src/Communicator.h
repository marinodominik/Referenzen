#ifndef AGENT_SRC_COMMUNICATOR_H_
#define AGENT_SRC_COMMUNICATOR_H_

#include <ostream>
using namespace std;

#include "udpsocket.h"
#include "basics/wmdef.h"
#include "basics/wmstate.h"

#include "behaviors/base_bm.h"

#include "MessagesInfo.h"
#include "CommandsInfo.h"

#define CMD_VIEW_HACK

class Communicator
{
private:

    UDPsocket sock;
    char buffer[ UDPsocket::MAXMESG ];

    bool got_sock_data;

#ifdef CMD_VIEW_HACK
    Cmd_View cmd_view_hack;
#endif

	Angle ang_LP_2_SRV_deg( Angle a );
	double x_LP_2_SRV( double x );
	double y_LP_2_SRV( double y );

public:

    Communicator();
    virtual ~Communicator();

    bool initialize();

#ifdef CMD_VIEW_HACK
    void set_cmd_view_hack(Cmd_View cmd_view);
#endif

    bool has_data_on_socket();
    void set_socket_nonblocking();

    int  idle(int ms_time, int s_time);

    int  receive_and_incorporate_server_messages( MessagesInfo & msg_info, CommandsInfo & cmd_info, WMstate * state, WMstate * state_full, BodyBehavior *body_controller, Cmd& last_cmd );

    void send_initialize_message();
    bool recv_initialize_message( PlayMode &pm );
    bool recv_parameter_messages();
    bool produce_parameter_messages( ostream &out );

    void send_cmd( Cmd const &cmd, Msg_teamcomm2 const &tc );
#ifdef CMD_VIEW_HACK
    void send_cmd_view( const Cmd_View & cmd_view );
#endif

    bool send_string_directly( string str );

};

#endif /* AGENT_SRC_COMMUNICATOR_H_ */
