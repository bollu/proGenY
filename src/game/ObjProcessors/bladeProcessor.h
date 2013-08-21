#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"

#include "../../core/Process/viewProcess.h"
#include "../../core/Process/worldProcess.h"

#include "../../util/mathUtil.h"


struct bladeData {};


class objectMgr;
class bladeProcessor : public objectProcessor
{
private:
	objectMgr *objectManager;


public:
	bladeProcessor ( processMgr &processManager, Settings &settings, eventMgr &_eventManager );
	void Process ( float dt );
	void postProcess ();
};