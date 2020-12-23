#ifndef __OVERCOME_OFFSIDE_08_WBALL_H__
#define __OVERCOME_OFFSIDE_08_WBALL_H__

#include "base_bm.h"
#include "Vector.h"
#include <math.h>

#include "../basics/Cmd.h"
#include "../basics/PlayerSet.h"
#include "skills/basic_cmd_bms.h"
#include "skills/oneortwo_step_kick_bms.h"
#include "skills/neuro_kick_wrapper_bms.h"
#include "skills/neuro_kick05_bms.h"
#include "globaldef.h"
#include "ws_info.h"
#include "ws.h"
#include "tools.h"
#include "ws_memory.h"
#include "overcome_offside_08_noball_bmc.h"


/* logging and drawing macros, enable for debugging! */
#if 1
#include "log_macros.h"
#define JTSPOL(XXX)   LOG_POL(0,<<"OOT08_WBALL: "<<XXX)
#define JTSPOL2(XXX)  LOG_POL(1,<<"OOT08_WBALL: "<<XXX)
#define MARK_POS(P,C) DRAW(VC2D((P),0.3,#C));
#define DRAW_LINE(P,Q,C) DRAW(VL2D((P),(Q),C))
#define DRAW(XXX)  LOG_POL(0,<<_2D<<XXX)
#define DRAW2(XXX) LOG_POL(1,<<_2D<<XXX)
#define TGoowLOGPOL(YYY,XXX)        LOG_POL(YYY,XXX)
#else
#define JTSPOL(XXX)
#define JTSPOL2(XXX)
#define MARK_POS(P,C)
#define DRAW_LINE(P,Q,C)
#define DRAW(XXX)  
#define DRAW2(XXX)
#define TGoowLOGPOL(YYY,XXX)  
#endif

/* RaumPassInfo struct
 * stores all Informations needed to execute a deadly pass
 */
struct RaumPassInfo {
  RaumPassInfo() {
    reset();
  };
  ~RaumPassInfo(){ };
  
  void reset() {
    first_kick.set = false;
    second_kick.set = false;
    isPassCommunicated = false;
    isPassChosen = false;
    receiver = NULL;
  };
  
  // some bools that five information about the pass state
  bool isPassChosen, isPassCommunicated;
  // detailed info for pass sequence
  struct simple_kick {
    ANGLE  ang;
    double vel;
    int    type;
    bool   set;
  };
  simple_kick first_kick; 
  simple_kick second_kick;  
  PPlayer     receiver;
};


class OvercomeOffside08Wball : public BodyBehavior {
	
	private:
    static const int     cvc_OOT_FUTURE_BALL_POSITIONS  = 40;
    static const double  cvc_INACCAPTABLE_OOT_PASS;
    static       Vector  cvPassBallPositionArray[cvc_OOT_FUTURE_BALL_POSITIONS];

		/* private variables */
		PlayerSet ivPassPlayers;
		RaumPassInfo *ivpRaumPass;
		OneOrTwoStepKick *ivpTwoStepKick;
    OneStepKick *ivpOneStepKick;
    NeuroKickWrapper *ivpNeuroKickWrapper;
    NeuroKick05 *ivpNeuroKick05;
		static bool initialized;
		double offside_line;
		double safety_margin;
		int offside_line_age;
		
		/* private methods */
		bool thereIsNoBetterPassComing(PPlayer bestPlayer, int startPos);
		bool passIsAbsolutelyDeadly(PPlayer bestPlayer);
		bool passWayCompletelyFree(PPlayer bestPlayer, const  Vector & ball_pos , double corrSize);
		bool getOptPassSequence( PPlayer bestPlayer, RaumPassInfo *lvPassSequence, double angle_diff);
		bool getPassSequence( PPlayer bestPlayer, RaumPassInfo *lvPassSequence, const ANGLE &passDir );
		bool getPassSequenceWithOneAdditionalKick(PPlayer bestPlayer, RaumPassInfo *lvPassSequence, const ANGLE &passDir);
		bool lookAheadPassWayStillFree(PPlayer bestPlayer, const Vector &ballPos, bool twoStepSet);
		bool followPassSequence(RaumPassInfo *passSequence, Cmd & cmd);
		void update_offside_line(double &offside_line, double &safetyMargin);
    bool announcePass(Cmd &cmd, RaumPassInfo *lvRaumPass );
		
    double evaluatePassRequestImportance( PPlayer   passRequester,
                                         int     & angleDelta,
                                         double  & passRelaxationSuggested );
    double evaluatePass( PPlayer passRequester, int passAngle );
    int getNumberOfKicksToPlayPass( double vel, int angInDegree,
                                    double &velAfter1Kick, double &velAfter2Kick );
    static double  evaluateOOTPassCorridor( PPlayer receiver,
                                           int    passAngle, //degree
                                           double offsideLine,
                                           int  & icptStepGuess );
    static void   fillOOTPassBallPositionsArray( Vector passStartPos,
                                                 int    passAngle );
    static int    getSmallestNumberOfInterceptionSteps( Vector  startPos,
                                                        int     passAngle,
                                                        double  offsideLine,
                                                        PPlayer receiver );

	public:
		
		/* constants */
		static const int    cvc_TWO_STEP;
		static const int    cvc_PREP_KICK ;
		static const int    cvc_PASS_ANGLE_SEARCH_DELTA;
		static const int    cvc_PASS_ANGLE_MAX_DEVIATION;
		static const double cvc_THREE_STEP_KEEPAWAY_CIRCLE;
		static const double cvc_MAX_PASS_VELOCITY_RELAXATION_UNDER_PRESSURE;
		
		/* public methods */
		OvercomeOffside08Wball();
		virtual ~OvercomeOffside08Wball();
		static bool init(char const * conf_file, int argc, char const* const* argv);
		bool get_cmd (Cmd & cmd);
		bool isOOTPassPossible();
    
    static int suggestAttentionToTeammateNumber( double   offsideLine );
	
};



#endif
