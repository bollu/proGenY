#pragma once
#include "../../core/Messaging/eventMgr.h"
#include "../ObjProcessors/groundMoveProcessor.h"


struct WSADHandlerData{
	sf::Keyboard::Key up;
	sf::Keyboard::Key down;
	sf::Keyboard::Key left;
	sf::Keyboard::Key right;

	moveData *objMoveData;
};

class WSADHandler : public Observer{
public:
	WSADHandler(eventMgr *_eventManager, WSADHandlerData WSADData) : 
	eventManager(_eventManager), WSADData(WSADData){
		
		eventManager->Register(Hash::getHash("keyPressed"), this);
		eventManager->Register(Hash::getHash("keyReleased"), this);
		
	};

	void recieveEvent(const Hash *eventName, baseProperty *eventData);
	void Update();

private:
	eventMgr *eventManager;
	WSADHandlerData WSADData;

	void _handleKeyPress(sf::Event::KeyEvent *event);
	void _handleKeyRelease(sf::Event::KeyEvent *event);
};
