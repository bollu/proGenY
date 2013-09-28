#pragma once
#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/eventMgr.h"
#include "../../core/componentSys/processor/objectProcessor.h"


#include "../../include/SFML/Graphics.hpp"
#include "../../core/math/vector.h"

#include "../../core/Rendering/viewProcess.h"


class terrainProcessor : public objectProcessor{
private:
	b2World *world;
	sf::RenderWindow *window;
	viewProcess *view;

public:
	terrainProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
		objectProcessor("terrainProcessor"){
	};
protected:
	bool _shouldProcess(Object *obj){
		return obj->hasProperty("terrainData");
	};

};
