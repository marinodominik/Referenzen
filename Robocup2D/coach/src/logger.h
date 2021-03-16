#include "options.h"
#include "coach.h"



#define LOGGING 1



#define BASIC_LOGGING 1


#if LOGGING && BASIC_LOGGING
#define LOG_DEF(LLL,XXX) LOG_TMP(LogOptions::xx_DEF,fld.getTime(),LLL,XXX)
#define LOG_FLD(LLL,XXX) LOG_TMP(LogOptions::xx_FLD,fld.getTime(),LLL,XXX)
#define LOG_ERR(LLL,XXX) LOG_TMP(LogOptions::xx_ERR,fld.getTime(),LLL,XXX)
#define LOG_VIS(LLL,XXX) LOG_TMP(LogOptions::xx_VIS,fld.getTime(),LLL,XXX)
#define LOG_MSG(LLL,XXX) LOG_TMP(LogOptions::xx_MSG,fld.getTime(),LLL,XXX)
#define LOG_INT(LLL,XXX) LOG_TMP(LogOptions::xx_INT,fld.getTime(),LLL,XXX)
#else
#define LOG_DEF(LLL,XXX) 
#define LOG_FLD(LLL,XXX) 
#define LOG_ERR(LLL,XXX) 
#define LOG_VIS(LLL,XXX) 
#define LOG_MSG(LLL,XXX) 
#define LOG_INT(LLL,XXX) 
#endif

//#define MSG_WITH_NUMBER(XXX)  " [ " << mdpInfo::mdp->me->player_number <<" ] " XXX
//#define MSG_WITH_NUMBER(XXX) " [ " << mdpInfo::mdp->me->number <<" ] " XXX

//#define LOG_ERR(LLL,XXX) LOG_TMP(LogOptions::xx_ERR,mdpInfo::mdp->time_current,LLL,MSG_WITH_NUMBER(XXX))
//#define LOG_DEB(LLL,XXX) LOG_TMP(LogOptions::xx_DEB,mdpInfo::mdp->time_current,LLL,XXX)

//#define LOG_WM_ERR(TIME,LLL,XXX) LOG_TMP(LogOptions::xx_ERR,TIME,LLL,XXX)
//#define LOG_WM(TIME,LLL,XXX) LOG_TMP(LogOptions::xx_WM,TIME,LLL,XXX);



#define LOG_TMP(WHICH,TIME,LLL,XXX) \
  if (LogOptions::max_level>= 0 && LLL <= LogOptions::max_level) { \
    if (LogOptions::opt_log[WHICH].log_cout) cout << "\n" << TIME << ".0 "  << LogOptions::indent[WHICH][LLL] << " " XXX << flush; \
    if (LogOptions::opt_log[WHICH].log_cerr) cerr << "\n" << TIME << ".0 " << LogOptions::indent[WHICH][LLL] << " "  XXX << flush; \
    if (LogOptions::opt_log[WHICH].log_file) LogOptions::file << "\n" << TIME << ".0 " << LogOptions::indent[WHICH][LLL] << " " XXX; }



#define _2D "_2D_ "

#define L2D(x1,y1,x2,y2,col)     " l "          << (x1) << " "  << (y1) << " " << (x2)  << " "   << (y2)  << " " << (col) << ";"
#define C2D(x1,y1,r,col)         " c "          << (x1) << " "  << (y1) << " " << (r)   << " "   << (col) << ";"
#define P2D(x1,y1,col)           " p "          << (x1) << " "  << (y1) << " " << (col) << ";"
#define STRING2D(x1,y1,text,col) " STRING col=" << col  << " (" << x1   << "," << y1    << ",\"" << text  << "\");"

#define VL2D(v1,v2,col)       " l "          << (v1).getX() << " "  << (v1).getY() << " " << (v2).getX() << " "   << (v2).getY() << " " << (col) << ";"
#define VC2D(v,r,col)         " c "          << (v).getX()  << " "  << (v).getY()  << " " << (r)         << " "   << (col)       << ";"
#define VP2D(v,col)           " p "          << (v).getX()  << " "  << (v).getY()  << " " << (col)       << ";"
#define VSTRING2D(v,text,col) " STRING col=" << col         << " (" << (v).getX()  << "," << (v).getY()  << ",\"" << text        << "\");"
