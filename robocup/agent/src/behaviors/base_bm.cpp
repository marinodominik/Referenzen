#include "base_bm.h"

BodyBehavior::BodyBehavior()
{

}

BodyBehavior::~BodyBehavior()
{

}

bool BodyBehavior::init( char const * conf_file, int argc, char const* const * argv )
{
    return true;
}

bool BodyBehavior::get_cmd( Cmd & cmd )
{
    return false;
}

void BodyBehavior::reset_intention()
{

}

char const* BodyBehavior::id()
{
    char const *name = typeid( *this ).name();

    while( ( *name > '9' || *name < '0' ) && *name != '\0' )
    {
        name++;
    }

    return ++name;
}

NeckBehavior::NeckBehavior()
{

}

NeckBehavior::~NeckBehavior()
{

}

bool NeckBehavior::init( char const* conf_file, int argc, char const* const* argv )
{
    return true;
}

bool NeckBehavior::get_cmd( Cmd &cmd )
{
    return false;
}

char const* NeckBehavior::id()
{
    char const* name = typeid( *this).name();

    while( ( *name > '9' || *name < '0' ) && *name != '\0' )
    {
        name++;
    }

    return ++name;
}

ViewBehavior::ViewBehavior()
{

}

ViewBehavior::~ViewBehavior()
{

}

bool ViewBehavior::init( char const* conf_file, int argc, char const* const* argv )
{
    return true;
}

bool ViewBehavior::get_cmd( Cmd &cmd )
{
    return false;
}

char const* ViewBehavior::id()
{
    char const* name = typeid( *this).name();

    while( ( *name > '9' || *name < '0' ) && *name != '\0' )
    {
        name++;
    }

    return ++name;
}

NeckViewBehavior::NeckViewBehavior()
{

}

NeckViewBehavior::~NeckViewBehavior()
{

}

bool NeckViewBehavior::init( char const* conf_file, int argc, char const* const* argv )
{
	return true;
}

bool NeckViewBehavior::get_cmd( Cmd &cmd )
{
	return false;
}

char const* NeckViewBehavior::id()
{
	char const* name = typeid( *this).name();

	while( ( *name > '9' || *name < '0' ) && *name != '\0' )
	{
		name++;
	}

	return ++name;
}

AttentionToBehavior::AttentionToBehavior()
{

}

AttentionToBehavior::~AttentionToBehavior()
{

}

bool AttentionToBehavior::init( char const* conf_file, int argc, char const* const* argv )
{
    return true;
}

bool AttentionToBehavior::get_cmd( Cmd &cmd )
{
    return false;
}

char const* AttentionToBehavior::id()
{
    char const* name = typeid( *this).name();

    while( ( *name > '9' || *name < '0' ) && *name != '\0' )
    {
        name++;
    }

    return ++name;
}

PointToBehavior::PointToBehavior()
{

}

PointToBehavior::~PointToBehavior()
{

}

bool PointToBehavior::init( char const* conf_file, int argc, char const* const* argv )
{
    return true;
}

bool PointToBehavior::get_cmd( Cmd &cmd )
{
    return false;
}

char const* PointToBehavior::id()
{
    char const* name = typeid( *this).name();

    while( ( *name > '9' || *name < '0' ) && *name != '\0' )
    {
        name++;
    }

    return ++name;
}
