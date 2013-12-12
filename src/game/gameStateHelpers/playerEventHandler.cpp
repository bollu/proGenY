#pragma once
#include "playerEventHandler.h"


playerEventHandler::playerEventHandler(eventMgr *_eventManager, playerHandlerData playerData) : 
eventManager(_eventManager), playerData(playerData){
	
	eventManager->Register(Hash::getHash("keyPressed"), this);
	eventManager->Register(Hash::getHash("keyReleased"), this);

	eventManager->Register(Hash::getHash("mouseWheelUp"), this);
	eventManager->Register(Hash::getHash("mouseWheelDown"), this);

	eventManager->Register(Hash::getHash("mouseMovedGame"), this);

	Object *player = playerData.player;


	assert(player != NULL);

	this->playerData.playerPos = player->getProp<vector2>(Hash::getHash("position"));
	this->playerData.objMoveData =  player->getProp<MoveData>(Hash::getHash("MoveData"));
	this->playerData.physicsData =  player->getProp<phyData>(Hash::getHash("phyData"));

	assert(this->playerData.playerPos != NULL);
	assert(this->playerData.objMoveData != NULL);
	assert(this->playerData.physicsData);
	this->firing = false;

};

void playerEventHandler::recieveEvent(const Hash *eventName, baseProperty *eventData){
	static const Hash *keyPressedHash = Hash::getHash("keyPressed");
	static const Hash *keyReleasedHash = Hash::getHash("keyReleased");
	static const Hash *mouseMovedGameHash = Hash::getHash("mouseMovedGame");

	static const Hash *mouseWheelUpHash = Hash::getHash("mouseWheelUp");
	static const Hash *mouseWheelDownHash = Hash::getHash("mouseWheelDown");


	if(eventName == mouseMovedGameHash){
		
		v2Prop *mousePos = dynamic_cast<v2Prop *>(eventData); 
		assert(mousePos != NULL);
		this->lastMousePos = *mousePos->getVal();

		
	}

	else if(eventName == keyPressedHash){
		Prop<sf::Event::KeyEvent> *eventProp = dynamic_cast< Prop<sf::Event::KeyEvent> *>(eventData); 

		
		assert(eventProp != NULL && "\nunable to receive event data\n");

		this->_handleKeyPress(eventProp->getVal());
	}
	else if(eventName == keyReleasedHash){

		Prop<sf::Event::KeyEvent> *eventProp = dynamic_cast< Prop<sf::Event::KeyEvent> *>(eventData); 

		assert(eventProp != NULL && "\nunable to receive event data\n");

		
		this->_handleKeyRelease(eventProp->getVal());

	}
	else if(eventName == mouseWheelUpHash){
		iProp *wheelDeltaProp = dynamic_cast< iProp *>(eventData); 
		int wheelDelta = *wheelDeltaProp->getVal();

		this->_handleMouseWheelUp(wheelDelta);
	}
	else if(eventName == mouseWheelDownHash){
		iProp *wheelDeltaProp = dynamic_cast< iProp *>(eventData); 
		int wheelDelta = *wheelDeltaProp->getVal();
		
		this->_handleMouseWheelDown(wheelDelta);
	}



};


void playerEventHandler::_handleKeyPress(sf::Event::KeyEvent *event){
	sf::Keyboard::Key key = event->code;

	if(key == this->playerData.up){
		playerData.objMoveData->Jump();
		
	}

	else if(key == this->playerData.left){
		playerData.objMoveData->setMoveLeft(true);
		playerData.objMoveData->setMoveRight(false);
		
	}

	else if(key == this->playerData.right){
		playerData.objMoveData->setMoveRight(true);
		playerData.objMoveData->setMoveLeft(false);
		
	}

	else if(key == this->playerData.fireGun){
		this->firing = true;
		
		
	}
};
void playerEventHandler::_handleKeyRelease(sf::Event::KeyEvent *event){

	sf::Keyboard::Key key = event->code;

	if(key == this->playerData.up){
	
	}

	if(key == this->playerData.left){
		playerData.objMoveData->setMoveLeft(false);
	}

	else if(key == this->playerData.right){
		playerData.objMoveData->setMoveRight(false);
	}

	else if(key == this->playerData.fireGun){
		this->firing = false;
		
	}
};



void playerEventHandler::_handleMouseWheelUp(int ticks){
	eventManager->sendEvent(Hash::getHash("nextGun"), ticks);
};

void playerEventHandler::_handleMouseWheelDown(int ticks){
	eventManager->sendEvent(Hash::getHash("prevGun"), ticks);
};

void playerEventHandler::Update(){

	phyData *phy = playerData.physicsData;

	this->_updateGunFacing(this->lastMousePos);

	for(collisionData collision : phy->collisions){
		if(collision.getCollidedObjectCollision() == Hash::getHash("terrain")){
			if(collision.type == collisionData::Type::onBegin){
				this->playerData.objMoveData->resetJump();
				break;
			}
		}
	}

	if(this->firing){
		this->playerData.player->sendMessage(Hash::getHash("fireGun"), true);
	};

};

void playerEventHandler::_updateGunFacing(vector2 gameMousePos){
	 

	vector2 delta = (gameMousePos - *this->playerData.playerPos).Normalize();
	util::Angle facing = util::Angle(delta);

	this->eventManager->sendEvent(Hash::getHash("playerFacingChanged"), facing);

	/*
	
	
	
	vector2 bulletOffset = facing.polarProjection(3);
	vector2 gunOffset = facing.polarProjection(0.2);

	playerData._GunData->setBulletPos(*playerData.playerPos + bulletOffset);
	playerData._GunData->setFacing(facing);

	playerData.gunOffsetData->posOffset = v
	*/
};

void playerEventHandler::_fireGun(){

	this->eventManager->sendEvent(Hash::getHash("firePlayerGun"));
	//playerData._GunData->Fire();
	
};
