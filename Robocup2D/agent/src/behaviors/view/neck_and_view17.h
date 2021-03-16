/*
 * NeckAndView17.h
 *
 *  Created on: Dec 18, 2016
 *      Author: romanber
 */

#ifndef NeckAndView17_H_
#define NeckAndView17_H_

//include some basic functionalities
#include "../base_bm.h"
#include "Cmd.h"
#include "log_macros.h"
#include "view/neck_cmd_bms.h"

// will probably be removed in the future:
#include "synch_view08_bmv.h"
#include "bs03_neck_bmn.h"

class NeckAndView17 : public NeckViewBehavior
{
private:
	SynchView08 *synch_view;
	BS03Neck *neck_view;
	//static NeckCmd *neck_cmd;

	static bool initialized;
	static bool is_between( ANGLE aim, ANGLE left, ANGLE right, int view_angle );
	static ANGLE isBetweenAngle( ANGLE aim, ANGLE left, ANGLE right, int view_angle, int desired_view = -1 );
	static ANGLE calculateTurn( double overall_angle, ANGLE expected_turn );
	static int urgent_timestep;

public:
	NeckAndView17();
	virtual ~NeckAndView17();

	bool get_cmd( Cmd &cmd );
	static bool init( char const * conf_file, int argc, char const* const* argv );

	/* make request method that is able to look behind the player if it is useful */
	static bool urgent_look_request( ANGLE angle, Cmd &cmd );
	static bool urgent_look_request( ANGLE nagle, int view, Cmd &cmd );
	static bool change_view ( int angle, Cmd &cmd );
};

#endif /* NeckAndView17_H_ */
