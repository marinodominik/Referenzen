#include "options.h"
#include "ws_info.h"



#define LOGGING 1



#define BASIC_LOGGING 1
#define ERROR_LOGGING 1
#define DEBUG_LOGGING 1
#define WORLD_MODEL_LOGGING 0


#if LOGGING && BASIC_LOGGING // activate in competitions only
#define LOG_MDP(LLL,XXX) LOG_TMP(LogOptions::xx_MDP,WSinfo::ws->time,LLL,XXX)
#define LOG_DEC(LLL,XXX) LOG_TMP(LogOptions::xx_DEF,WSinfo::ws->time,LLL,XXX)
#define LOG_MOV(LLL,XXX) LOG_TMP(LogOptions::xx_DEF,WSinfo::ws->time,LLL,XXX)
#define LOG_POL(LLL,XXX) LOG_TMP(LogOptions::xx_DEF,WSinfo::ws->time,LLL,XXX)
#define LOG_STR(LLL,XXX) LOG_TMP(LogOptions::xx_DEF,WSinfo::ws->time,LLL,XXX)
#else
#define LOG_MDP(LLL,XXX)
#define LOG_DEC(LLL,XXX)
#define LOG_MOV(LLL,XXX)
#define LOG_POL(LLL,XXX)
#define LOG_STR(LLL,XXX)
#endif

#if LOGGING && ERROR_LOGGING
#define LOG_ERR(LLL,XXX) LOG_TMP(LogOptions::xx_ERR,WSinfo::ws->time,LLL,MSG_WITH_NUMBER(XXX))
#else
#define LOG_ERR(LLL,XXX)
#endif

#if LOGGING && DEBUG_LOGGING
#define LOG_DEB(LLL,XXX) LOG_TMP(LogOptions::xx_DEB,WSinfo::ws->time,LLL,XXX)
//#define LOG_DEB(LLL,XXX) LOG_TMP(LogOptions::xx_DEB,WSinfo::ws->time << ".0 - (" << __FILE__ << __LINE__ << ")",LLL,XXX)
#else
#define LOG_DEB(LLL,XXX)
#endif

#if LOGGING && WORLD_MODEL_LOGGING
#define LOG_WM(TIME,LLL,XXX) LOG_TMP(LogOptions::xx_WM,TIME,LLL,XXX);
#define LOG_WM_APPEND(LLL,XXX) LOG_TMP_APPEND(LogOptions::xx_WM,LLL,XXX);
#define LOG_WM_ERR(TIME,LLL,XXX) LOG_TMP(LogOptions::xx_ERR,TIME,LLL,XXX)
#else
#define LOG_WM(TIME,LLL,XXX)
#define LOG_WM_APPEND(LLL,XXX)
#define LOG_WM_ERR(TIME,LLL,XXX)
#endif



#define MSG_WITH_NUMBER(XXX) " [ " << ClientOptions::player_no <<" ] " XXX



#define LOG_TMP(WHICH,TIME,LLL,XXX) \
		if (LogOptions::max_level>= 0 && LLL <= LogOptions::max_level) { \
			if (LogOptions::opt_log[WHICH].log_cout) std::cout << "\n" << (WSinfo::ws?TIME:0) << ".0 "  << LogOptions::indent[WHICH][LLL] << " " XXX ; \
			if (LogOptions::opt_log[WHICH].log_cerr) std::cerr << "\n" << (WSinfo::ws?TIME:0) << ".0 " << LogOptions::indent[WHICH][LLL] << " "  XXX ; \
			if (LogOptions::opt_log[WHICH].log_file) LogOptions::file << "\n" << (WSinfo::ws?TIME:0) << ".0 " << LogOptions::indent[WHICH][LLL] << " " XXX; }

//don't use new line, just append text (useful in loops; always use at least one LOG_TMP before)!
#define LOG_TMP_APPEND(WHICH,LLL,XXX) \
		if (LogOptions::max_level>= 0 && LLL <= LogOptions::max_level) { \
			if (LogOptions::opt_log[WHICH].log_cout) std::cout << XXX ; \
			if (LogOptions::opt_log[WHICH].log_cerr) std::cerr << XXX ; \
			if (LogOptions::opt_log[WHICH].log_file) LogOptions::file << XXX; }



#define _2D "_2D_ "

#define L2D(x1,y1,x2,y2,col)   " l "          << (x1)        << " "  << (y1)        << " " << (x2)        << " "   << (y2)        << " " << (col) << ";"
#define C2D(x,y,r,col)         " c "          << (x)         << " "  << (y)         << " " << (r)         << " "   << (col)       << ";"
#define P2D(x,y,col)           " p "          << (x)         << " "  << (y)         << " " << (col)       << ";"
#define STRING2D(x,y,text,col) " STRING col=" << col         << " (" << x           << "," << y           << ",\"" << text        << "\");"

#define VL2D(v1,v2,col)        " l "          << (v1).getX() << " "  << (v1).getY() << " " << (v2).getX() << " "   << (v2).getY() << " " << (col) << ";"
#define VC2D(v,r,col)          " c "          << (v).getX()  << " "  << (v).getY()  << " " << (r)         << " "   << (col)       << ";"
#define VP2D(v,col)            " p "          << (v).getX()  << " "  << (v).getY()  << " " << (col)       << ";"
#define VSTRING2D(v,text,col)  " STRING col=" << col         << " (" << (v).getX()  << "," << (v).getY()  << ",\"" << text        << "\");"
