/*
 * NeckAndView17.cpp
 *
 *  Created on: Dec 18, 2016
 *      Author: romanber
 */

#include "neck_and_view17.h"

#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"
#include "blackboard.h"
#include "Cmd.h"

#include <iostream>

// will probably be removed in the future:
#include "synch_view08_bmv.h"

int NeckAndView17::urgent_timestep = -1;

/*
 * This is an attempt to combine the neck and view behavior in a single class
 *
 * 1. try to use this behavior - done
 * 2.1 let all players look to the ball - done
 * 2.2 do nothing to view behavior at this point
 * 3.1 review code of old neck behavior
 * 3.2 requests
 * 4. win all games and receive title
 */

bool NeckAndView17::initialized = false;
//NeckCmd NeckAndView17::neck_cmd;

NeckAndView17::NeckAndView17()
{
	synch_view = new SynchView08();
	neck_view = new BS03Neck();
	//neck_cmd = new NeckCmd();
}


NeckAndView17::~NeckAndView17()
{
	delete synch_view;
	delete neck_view;
	//delete neck_cmd;
}


// this method sets is in charge of executing view and neck commands
// 07.06.17 - why is this needed again?
bool NeckAndView17::get_cmd( Cmd &cmd )
{
	// FIRST STAGE: all players should look to the ball

	// the players should look to the ball (neck part)
	//look_to_ball( cmd );
	LOG_POL( 3, << "NeckAndView17: neck_view");

	if( NeckAndView17::urgent_timestep != WSinfo::me->time )
	{
		// do BS03Neck
		LOG_POL( 3, "NeckAndView17: do \"normal\" view behavior" );
		neck_view->get_cmd( cmd );

	// do BS03Neck
	LOG_POL( 3, "NeckAndView17: do \"normal\" view behavior" );
	neck_view->get_cmd( cmd );

	// do SynchView08
	LOG_POL( 3, << "NeckAndView17: synch_view" );
	synch_view->get_cmd( cmd );


		// do SynchView08
		LOG_POL( 3, << "NeckAndView17: synch_view" );
		synch_view->get_cmd( cmd );
	}

	// this is the "normal" view behavior at this time - used as placeholder until
	// a the new behavior has progressed to a point to replace this

	return true;
}


// init-method (prints error to log if something went wrong)
bool NeckAndView17::init( char const * conf_file, int argc, char const* const* argv )
{
	LOG_POL(3, "NeckAndView17 initialized");

	if (initialized)
	{
		return initialized;
	}

	initialized = true;

	if (!initialized)
	{
		LOG_POL(3, "NeckAndView17 not initialized!");
	}

	return initialized;
}

// urgent_look_request calculates the best fitting view angle and neck orientation to match a
// given absolute angle on the playfield.
bool NeckAndView17::urgent_look_request( ANGLE target_angle, Cmd &cmd )
{
	NeckAndView17::urgent_timestep = WSinfo::me->time;
	// determine player's current view (narrow = 0, normal = 1, wide = 2)

	Vector from_target = Vector( target_angle ) + WSinfo::me->pos;
	LOG_POL( 3, << "x: " << from_target.getX() << ", y:" << from_target.getY() );

	int view = WSinfo::ws->view_angle;

	ANGLE expected_turn_angle = Tools::my_expected_turn_angle();

	// determine player's current absolute angle and his neck turn boundaries
	ANGLE current_angle = WSinfo::me->ang + expected_turn_angle;
	ANGLE player_left_max = ANGLE( current_angle + 0.5 * PI );
	ANGLE player_right_max = ANGLE( current_angle - 0.5 * PI );


	ANGLE current_angle_mod = WSinfo::me->ang;
	ANGLE player_left_max_mod = ANGLE( current_angle_mod + 0.5 * PI );
	ANGLE player_right_max_mod = ANGLE( current_angle_mod - 0.5 * PI );


	double view_angle_mod = (double)view + 1.0;
	ANGLE angle_mod2 = ANGLE( view_angle_mod * 0.5235987755982989 );

	LOG_POL( 3, << _2D << VL2D( WSinfo::me->pos, Vector( WSinfo::me->pos + Vector( WSinfo::me->ang - DEG2RAD( 90.0 ) - angle_mod2 ).normalize( 5.0 ) ), "f679da" ) );
	LOG_POL( 3, << _2D << VL2D( WSinfo::me->pos, Vector( WSinfo::me->pos + Vector( WSinfo::me->ang + DEG2RAD( 90.0 ) + angle_mod2 ).normalize( 5.0 ) ), "f679da" ) );

	LOG_POL( 3, << _2D << VL2D( WSinfo::me->pos, Vector( WSinfo::me->pos + Vector( current_angle - DEG2RAD( 90.0 ) - angle_mod2 ).normalize( 5.0 ) ), "09ff00" ) ) ;
	LOG_POL( 3, << _2D << VL2D( WSinfo::me->pos, Vector( WSinfo::me->pos + Vector( current_angle + DEG2RAD( 90.0 ) + angle_mod2 ).normalize( 5.0 ) ), "09ff00" ) ) ;

	// determine player's current neck angle
	ANGLE current_neck_angle = WSinfo::me->neck_ang;

	ANGLE test_target = Tools::my_neck_angle_to( from_target );
	LOG_ERR( 0, << "Angle to target: " << Tools::radianInDegree( test_target.get_value() ));

	LOG_POL( 3, << "Expected Turn Angle: " << Tools::my_expected_turn_angle().get_value_mPI_pPI() );

	double neck_angle_rel = RAD2DEG( WSinfo::me->neck_ang_rel.get_value_mPI_pPI() );

	int change_view_to;
	ANGLE set_turn_to;

	double overall_angle = neck_angle_rel + test_target.get_value_mPI_pPI() * (180.0/PI) - expected_turn_angle.get_value_mPI_pPI();
	double expected_turn = test_target.get_value_mPI_pPI() - expected_turn_angle.get_value_mPI_pPI();

	LOG_POL( 3, << "NeckAndView17: neck angle rel   deg: " << neck_angle_rel );
	LOG_POL( 3, << "NeckAndView17: neck angle rel      : " << Tools::degreeInRadian( neck_angle_rel ) );
	LOG_POL( 3, << "NeckAndView17: neck angle       deg: " << Tools::radianInDegree( WSinfo::me->neck_ang ) );
	LOG_POL( 3, << "NeckAndView17: neck angle          : " << WSinfo::me->neck_ang );
	LOG_POL( 3, << "NeckAndView17: target angle     deg: " << Tools::radianInDegree( target_angle ) );
	LOG_POL( 3, << "NeckAndView17: target angle        : " << target_angle );
	LOG_POL( 3, << "NeckAndView17: my_neck_angle_to deg: " << (test_target.get_value_mPI_pPI() * (180.0/PI)) );
	LOG_POL( 3, << "NeckAndView17: my_neck_angle_to    : " << test_target.get_value_mPI_pPI() );
	LOG_POL( 3, << "NeckAndView17: combined_angle   deg: " << overall_angle );

	// look if target is reachable with _current_ view angle and neck turning
	if ( is_between( target_angle, player_left_max, player_right_max, view ) )
	{
		LOG_POL( 3, << "NeckAndView17: target is in between" );
		LOG_POL( 3, << "NeckAndView17: [branch1] decision: set_turn( " << set_turn_to << " )" );

		set_turn_to = calculateTurn( overall_angle, expected_turn );
	}
	else
	{
		LOG_ERR( 0, << "target is NOT in between" );
		LOG_POL( 3, << "NeckAndView17: Check which view angle is the best one" );

		if ( is_between( target_angle, player_left_max, player_right_max, (view+1) ) )
		{
			if ( view+1 <= Cmd_View::VIEW_ANGLE_WIDE )
			{
				change_view_to = view+1;
				set_turn_to = calculateTurn( overall_angle, expected_turn );
				LOG_POL( 3, << "NeckAndView17: [branch2] decision: set_turn( " << set_turn_to << " )" );
			}
		}
		else if ( is_between( target_angle, player_left_max, player_right_max, (view+2) ) )
		{
			if ( view+2 <= Cmd_View::VIEW_ANGLE_WIDE )
			{
				change_view_to = view+2;
				set_turn_to = calculateTurn( overall_angle, expected_turn );
				LOG_POL( 3, << "NeckAndView17: [branch3] decision: set_turn( " << set_turn_to << " )" );
			}
		}

		change_view( change_view_to, cmd );
	}

	cmd.cmd_neck.set_turn( set_turn_to );

	return false;
}

// This method is an overloaded version of the urgent_look_request that allows to specifiy
// a predetermined resulting view angle of a player.
bool NeckAndView17::urgent_look_request( ANGLE target_angle, int view, Cmd &cmd )
{
	NeckAndView17::urgent_timestep = WSinfo::me->time;

	ANGLE expected_turn_angle = Tools::my_expected_turn_angle();
	Vector from_target = Vector( target_angle ) + WSinfo::me->pos;
	ANGLE test_target = Tools::my_neck_angle_to( from_target );

	double neck_angle_rel = RAD2DEG( WSinfo::me->neck_ang_rel.get_value_mPI_pPI() );

	ANGLE set_turn_to;

	double overall_angle = neck_angle_rel + test_target.get_value_mPI_pPI() * (180.0/PI);
	double expected_turn = test_target.get_value_mPI_pPI() - expected_turn_angle.get_value_mPI_pPI();

	ANGLE current_angle = WSinfo::me->ang + expected_turn_angle;
	ANGLE player_left_max = ANGLE( current_angle + 0.5 * PI );
	ANGLE player_right_max = ANGLE( current_angle - 0.5 * PI );

	if ( is_between( target_angle, player_left_max, player_right_max, view ) )
	{
		LOG_POL( 3, << "NeckAndView17: target is in between" );
		LOG_POL( 3, << "NeckAndView17: [branch1] decision: set_turn( " << set_turn_to << " )" );

		set_turn_to = calculateTurn( overall_angle, expected_turn );
	}

	change_view( view, cmd );
	cmd.cmd_neck.set_turn( set_turn_to );

	return false;
}


ANGLE NeckAndView17::calculateTurn( double turn_angle, ANGLE expected_turn )
{
	if( turn_angle < -180.0 )
	{
		LOG_POL( 3, << "NeckAndView17: -PI" );
		return Tools::degreeInRadian( -180.0 );
	}
	else if( turn_angle > 180.0 )
	{
		LOG_POL( 3, << "NeckAndView17: +PI" );
		return Tools::degreeInRadian( 180.0 );
	}
	else
	{
		LOG_POL( 3, << "NeckAndView17: NOT -PI or +PI" );
		return ANGLE( expected_turn );
	}
}

// this method is used for setting the view either to narrow, normal, or wide during runtime
bool NeckAndView17::change_view( int angle, Cmd &cmd )
{
	switch( angle )
	{
	case Cmd_View::VIEW_ANGLE_NARROW:
		cmd.cmd_view.set_angle_and_quality( Cmd_View::VIEW_ANGLE_NARROW, Cmd_View::VIEW_QUALITY_HIGH );
		return true;
	case Cmd_View::VIEW_ANGLE_NORMAL:
		cmd.cmd_view.set_angle_and_quality( Cmd_View::VIEW_ANGLE_NORMAL, Cmd_View::VIEW_QUALITY_HIGH );
		return true;
	case Cmd_View::VIEW_ANGLE_WIDE:
		cmd.cmd_view.set_angle_and_quality( Cmd_View::VIEW_ANGLE_WIDE, Cmd_View::VIEW_QUALITY_HIGH );
		return true;
	}

	return false;
}


// something seems to be wrong with this method but for testing purposes it will fit
// will review it later
bool NeckAndView17::is_between( ANGLE aim, ANGLE left, ANGLE right, int view_angle )
{
	double view_angle_mod = (double)view_angle + 1.0;
	ANGLE angle_mod = ANGLE( view_angle_mod * 0.5235987755982989 );

	ANGLE left_mod = ANGLE( left + angle_mod );
	ANGLE right_mod = ANGLE( right - angle_mod );

	ANGLE diff_1 = left_mod.diff( aim );
	ANGLE diff_2 = right_mod.diff( aim );

	if ( ( diff_1.get_value() + diff_2.get_value() ) <= ( 0.000005 + PI - 2.0 * angle_mod.get_value() ) )
	{
		return false;
	}

	return true;
}




