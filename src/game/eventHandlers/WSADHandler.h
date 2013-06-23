#pragma once
#include "../../core/Messaging/eventMgr.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/gunProcessor.h"
#include "../ObjProcessors/offsetProcessor.h"

struct WSADHandlerData{
public:
	sf::Keyboard::Key up;
	sf::Keyboard::Key down;
	sf::Keyboard::Key left;
	sf::Keyboard::Key right;
	sf::Keyboard::Key fireGun;
	
	Object *currentGun;
	Object *player;
private:
	friend class WSADHandler;

	vector2 *playerPos;
	moveData *objMoveData;
	phyData *physicsData;

	offsetData *gunOffsetData;
	gunData *_gunData;

};

class WSADHandler : public Observer{
public:
	WSADHandler(eventMgr *_eventManager, WSADHandlerData WSADData) : 
	eventManager(_eventManager), WSADData(WSADData){
		
		eventManager->Register(Hash::getHash("keyPressed"), this);
		eventManager->Register(Hash::getHash("keyReleased"), this);


		eventManager->Register(Hash::getHash("mouseMovedGame"), this);
	
		Object *player = WSADData.player;
		Object *gun = WSADData.currentGun;

		this->WSADData.playerPos = player->getProp<vector2>(Hash::getHash("position"));
		this->WSADData.objMoveData =  player->getProp<moveData>(Hash::getHash("moveData"));
		this->WSADData.physicsData =  player->getProp<phyData>(Hash::getHash("phyData"));
		this->WSADData.gunOffsetData = gun->getProp<offsetData>(Hash::getHash("offsetData"));
		this->WSADData._gunData = gun->getProp<gunData>(Hash::getHash("gunData"));
	};

	void recieveEvent(const Hash *eventName, baseProperty *eventData);
	void Update();

private:
	eventMgr *eventManager;
	WSADHandlerData WSADData;

	void _handleKeyPress(sf::Event::KeyEvent *event);
	void _handleKeyRelease(sf::Event::KeyEvent *event);

	void _updateGunFacing(vector2 gameMousePos);
	void _fireGun();
};
