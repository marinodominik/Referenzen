#ifndef _OO_TRAP08_NOBALL_BMC_H_
#define _OO_TRAP08_NOBALL_BMC_H_

#include "../basics/Cmd.h"
#include "base_bm.h"
#include "log_macros.h"
#include "tools.h"
#include "ws_info.h"


/* logging and drawing macros, enable for debugging! */
#if 1
#define   TGooLOGPOL(YYY,XXX)        LOG_POL(YYY,XXX)
#else
#define   TGooLOGPOL(YYY,XXX)
#endif

class OvercomeOffside08Noball : public BodyBehavior 
{
  public:
    /* public variables */
    static const int     cvc_OOT_FUTURE_BALL_POSITIONS  = 50;//TG09: erhoeht von 40
    static const double  cvc_ASSUMED_OOT_PASS_SPEED;
    static const double  cvc_INACCAPTABLE_OOT_PASS;
	
	private:
	
		/* private variables */
		static       bool    cvInitialized;
    static       Vector  cvPassBallPositionArray[cvc_OOT_FUTURE_BALL_POSITIONS];
    static       PPlayer cvPassReceivingPlayer;
    static       long    cvTimeOfRecentPassRequest;
    static       int     cvDegAngleOfRecentPassRequest;
			
    static Vector estimateOOTPassStartPosition( Vector startPlayerPos,
                                                double offsideLine );
    static void   fillOOTPassBallPositionsArray( Vector passStartPosition,
                                                 int    passAngle );
    static int    getSmallestNumberOfInterceptionSteps( Vector passStartPosition,
                                                        int    passAngle,
                                                        double offsideLine );

  public:
		
		static bool init(char const * conf_file, int argc, char const* const* argv) 
    {
			if (cvInitialized) 
				return true;
			cvInitialized = true;
			return cvInitialized;
		};
		
    //offside line should NOT be the real offside line, but rather a
    //more precautious one (e.g. ivCriticalOffsideLine from Noball05)
    static bool   determineBestOOTPassCorridor( PPlayer passStartPlayer,
                                                PPlayer passReceivingPlayer,
                                                double  offsideLine,
                                                int   & bestAngleParameter,
                                                Vector& passInterceptionPoint);
    static double evaluateOOTPassCorridor( Vector passStartPosition,
                                           int    passAngle, //degree
                                           double offsideLine,
                                           int  & icptStepGuess );
    static bool   isPassTotallyInacceptable( Vector  passStartPosition,   
                                             PPlayer passReceivingPlayer, 
                                             double  offsideLine,
                                              int     passAngle); //in degree, e.g. -30 or 50
    static bool   isPassTotallyInacceptable( PPlayer passStartPlayer,    
                                             PPlayer passReceivingPlayer, 
                                             double  offsideLine,
                                             int     passAngle );
    static void   setPassReceivingPlayer( PPlayer p );
};


#endif
