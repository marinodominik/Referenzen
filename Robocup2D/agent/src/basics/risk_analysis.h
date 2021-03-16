/*
 * risk_analysis.h
 *
 *  Created on: 09.12.2015
 *      Author: tobias und sefer
 */

#ifndef BASICS_RISK_ANALYSIS_H_
#define BASICS_RISK_ANALYSIS_H_

#include "globaldef.h"
#include "ws.h"
#include "log_macros.h"
#include "shadow.h"

class Risk_Analysis
{
public:
	int get_risk(const Ball *ball, const Player *goalie, int depth = 1);	//Gibt eine Risiko Bewertung zwischen 0 und 2 zurück
															// 0 - 	sicher
															// 1 - 	Kick nicht durchführbar
															// 2 -	in catch area
															//-1 -	Fehler
	Risk_Analysis();
	virtual ~Risk_Analysis();
};

#endif /* BASICS_RISK_ANALYSIS_H_ */
