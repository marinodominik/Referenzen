/*
 * risk_analysis.cpp
 *
 *  Created on: 09.12.2015
 *      Author: tobias und sefer
 */

#include "risk_analysis.h"

Risk_Analysis::Risk_Analysis()
{
	// TODO Auto-generated constructor stub

}

Risk_Analysis::~Risk_Analysis()
{
	// TODO Auto-generated destructor stub
}

int Risk_Analysis::get_risk(const Ball *ball, const Player *goalie, int depth)
{
	int return_value;
	Shadow shadow = Shadow();
	LOG_POL(3, <<"ANALYSIS FOLLOWING");
	double distance;

	if(depth < 0)
	{
		LOG_POL(1, << "Can't predict past");
		return -1;
	}

	if(ball == NULL)
	{
		LOG_POL(1, << "Risk Analysis FAILED! REASON: ball Object was NULL");
		return -1;
	}

	if(goalie == NULL)
	{
		LOG_POL(3, << "Risk Analysis has no intel about opponent's goalie. As he is not known, SAFETY is ASSUMED");
		return 0;
	}

	if(goalie->tackle_flag)
	{
		LOG_POL(3, << "Goalie tackled!");
		return 0;
	}

	shadow.set_goalie(goalie->pos);
	shadow.set_goal(0);
	shadow.set_point(ball->pos);
	shadow.set_prediction_depth(depth);

	distance = ball->pos.distance(goalie->pos);

	if(distance <= shadow.get_goalie_intercept_distance(depth))
	{
		return_value = 2;
	}
	else if(distance <= shadow.get_goalie_intercept_distance(depth + 3))
	{
		return_value = 1;
	}
	else
	{
		return_value = 0;
	}

	return return_value;
}
