#pragma once
#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"
#include "../../core/objectProcessor.h"


#include "../../include/SFML/Graphics.hpp"
#include "../../core/vector.h"

#include "../../core/Process/viewProcess.h"


class terrainProcessor : public objectProcessor{
private:
	b2World *world;
	sf::RenderWindow *window;
	viewProcess *view;

public:
	terrainProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager){
		
	};


	void Process(float dt){};
};