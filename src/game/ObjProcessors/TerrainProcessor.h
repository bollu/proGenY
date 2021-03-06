#pragma once
#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"
#include "../../core/componentSys/processor/ObjectProcessor.h"


#include "../../include/SFML/Graphics.hpp"
#include "../../core/math/vector.h"

#include "../../core/Rendering/viewProcess.h"


class terrainProcessor : public ObjectProcessor{
private:
	b2World *world;
	//sf::RenderWindow *window;
	//viewProcess *view;

public:
	terrainProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
		ObjectProcessor("terrainProcessor"){
	};
protected:
	bool _shouldProcess(Object *obj){
		return obj->hasProperty("terrainData");
	};

};
