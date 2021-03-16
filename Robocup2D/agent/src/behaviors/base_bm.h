#ifndef _BASE_BM_H_
#define _BASE_BM_H_

#include <typeinfo>
#include "../basics/Cmd.h"

class BodyBehavior
{
public:
    BodyBehavior();
    virtual ~BodyBehavior();

    static bool init( char const *conf_file, int argc, char const* const* argv );

    virtual bool get_cmd( Cmd &cmd );

    virtual void reset_intention();

    /** don't overwrite this id() method,
     it will deliver the class names also in child classes */
    virtual char const* id();
};

class NeckBehavior
{
public:
    NeckBehavior();
    virtual ~NeckBehavior();

    static bool init( char const* conf_file, int argc, char const* const* argv );

    virtual bool get_cmd( Cmd &cmd );

    /** don't overwrite this id() method,
     it will deliver the class names also in child classes */
    virtual char const* id();
};

class ViewBehavior
{
public:
    ViewBehavior();
    virtual ~ViewBehavior();

    static bool init( char const* conf_file, int argc, char const* const* argv );

    virtual bool get_cmd( Cmd &cmd );

    /** don't overwrite this id() method,
     it will deliver the class names also in child classes */
    virtual char const* id();
};


class NeckViewBehavior
{
public:
	NeckViewBehavior();
	virtual ~NeckViewBehavior();

	static bool init( char const* conf_file, int argc, char const* const* argv );

	virtual bool get_cmd( Cmd &cmd );

	/** don't overwrite this id() method,
	it will deliver the class names also in child classes */
	virtual char const* id();
};


class AttentionToBehavior
{
public:
    AttentionToBehavior();
    virtual ~AttentionToBehavior();

    static bool init( char const* conf_file, int argc, char const* const* argv );

    virtual bool get_cmd( Cmd &cmd );

    /** don't overwrite this id() method,
     it will deliver the class names also in child classes */
    virtual char const* id();
};

class PointToBehavior
{
public:
    PointToBehavior();
    virtual ~PointToBehavior();

    static bool init( char const* conf_file, int argc, char const* const* argv );

    virtual bool get_cmd( Cmd &cmd );

    /** don't overwrite this id() method,
     it will deliver the class names also in child classes */
    virtual char const* id();
};

#endif
