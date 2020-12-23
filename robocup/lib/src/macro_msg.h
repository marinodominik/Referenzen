#ifndef _MACRO_MSG_H_
#define _MACRO_MSG_H_

#include <iostream>

#include <fstream>
extern std::ofstream gvDummyOutputStream;

#include "../../agent/src/basics/WM.h"

#ifndef ERROR_STREAM
//#define ERROR_STREAM std::cerr
#define ERROR_STREAM gvDummyOutputStream
#endif

#ifndef ERROR_OUT
#define ERROR_OUT ERROR_STREAM << "\n\n*** ERROR file=\"" << __FILE__ << "\" line=" << __LINE__
#endif

#ifndef WARNING_STREAM
//#define WARNING_STREAM std::cerr
#define WARNING_STREAM gvDummyOutputStream
#endif

#ifndef WARNING_OUT
#define WARNING_OUT WARNING_STREAM << "\n\n*** WARNING file=\"" << __FILE__ << "\" line=" << __LINE__
#endif

#ifndef INFO_STREAM
//#define INFO_STREAM std::cout
#define INFO_STREAM gvDummyOutputStream
#endif

#ifndef INFO_OUT
#define INFO_OUT INFO_STREAM << "\n\n*** INFO file=\"" << __FILE__ << "\" line=" << __LINE__
#endif

#ifndef ID
#define ID " (#" << WM::my_number << ", time=" << WM::time << ") "
#endif


#endif
