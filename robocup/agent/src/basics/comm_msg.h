#ifndef _COMM_MSG_H_
#define _COMM_MSG_H_

struct SayMsg {
    SayMsg()
    {
        valid = false;
        time = param1 = param2 = -1;
        from = type = 0;
    }
    bool valid;
    int from;
    int time;
    unsigned char type;
    short param1;
    short param2;
};

#endif
