#pragma once
#include "../../core/controlFlow/eventMgr.h"
#include "../../core/componentSys/processor/phyProcessor.h"
#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/gunProcessor.h"
#include "../ObjProcessors/offsetProcessor.h"

struct playerHandlerData{
public:
	sf::Keyboard::Key up;
	sf::Keyboard::Key down;
	sf::Keyboard::Key left;
	sf::Keyboard::Key right;
	sf::Keyboard::Key fireGun;
	
	//Object *currentGun;
	Object *player;
private:
	friend class playerEventHandler;

	vector2 *playerPos;
	groundMoveData *objMoveData;
	phyData *physicsData;

};

class playerEventHandler : public Observer{
public:
	playerEventHandler(eventMgr *_eventManager, playerHandlerData playerData);
	void recieveEvent(const Hash *eventName, baseProperty *eventData);
	void Update();

private:
	eventMgr *eventManager;
	playerHandlerData playerData;


	//the last known mouse position
	vector2 lastMousePos;
	bool firing;


	void _handleKeyPress(sf::Event::KeyEvent *event);
	void _handleKeyRelease(sf::Event::KeyEvent *event);

	void _handleMouseWheelUp(int ticks);
	void _handleMouseWheelDown(int ticks);
	
	void _updateGunFacing(vector2 gameMousePos);
	void _fireGun();
};
