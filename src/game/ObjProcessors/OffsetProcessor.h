#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/math/vector.h"
#include "../../include/SFML/Graphics.hpp"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"

#include "../../core/Rendering/viewProcess.h"
#include "../../core/World/worldProcess.h"
#include "../../core/math/mathUtil.h"


struct OffsetData{
public:
	vector2 posOffset;
	util::Angle angleOffset;

	bool offsetPos;
	bool offsetAngle;

	OffsetData() : offsetPos(true), offsetAngle(true){}
};


class OffsetProcessor : public ObjectProcessor{
public:
	OffsetProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
	ObjectProcessor("OffsetProcessor"){
	}

protected:
	void _Process(Object *obj, float dt);
	bool _shouldProcess(Object *obj);
};
