#pragma once
#include "WSADHandler.h"

void WSADHandler::recieveEvent(const Hash *eventName, baseProperty *eventData){
	static const Hash *keyPressedHash = Hash::getHash("keyPressed");
	static const Hash *keyReleasedHash = Hash::getHash("keyReleased");

	if(eventName == keyPressedHash){
		Prop<sf::Event::KeyEvent> *eventProp = dynamic_cast< Prop<sf::Event::KeyEvent> *>(eventData); 

		

		assert(eventProp != NULL && "\nunable to receive event data\n");

		this->_handleKeyPress(eventProp->getVal());
	}
	else if(eventName == keyReleasedHash){

		Prop<sf::Event::KeyEvent> *eventProp = dynamic_cast< Prop<sf::Event::KeyEvent> *>(eventData); 

		assert(eventProp != NULL && "\nunable to receive event data\n");

		
		this->_handleKeyRelease(eventProp->getVal());

	}
};


void WSADHandler::_handleKeyPress(sf::Event::KeyEvent *event){
	sf::Keyboard::Key key = event->code;

	if(key == this->WSADData.up){
		WSADData.objMoveData->Jump();
		
	}

	if(key == this->WSADData.left){
		WSADData.objMoveData->setMoveLeft(true);
		WSADData.objMoveData->setMoveRight(false);
		
	}

	if(key == this->WSADData.right){
		WSADData.objMoveData->setMoveRight(true);
		WSADData.objMoveData->setMoveLeft(false);
		
	}
};
void WSADHandler::_handleKeyRelease(sf::Event::KeyEvent *event){

	sf::Keyboard::Key key = event->code;

	if(key == this->WSADData.up){
	
	}

	if(key == this->WSADData.left){
		WSADData.objMoveData->setMoveLeft(false);
	}

	if(key == this->WSADData.right){
		WSADData.objMoveData->setMoveRight(false);
	}
};

void WSADHandler::Update(){


	moveData *data = WSADData.objMoveData;
	
	if(!data->isMovingLeft() && !data->isMovingRight() && !data->isMidJump()){
		data->setMovementHalt(true);
		
	}else{
		data->setMovementHalt(false);
	}

	phyData *phy = WSADData.physicsData;

	
	for(collisionData collision : phy->collisions){
		if(collision.data->collisionType == Hash::getHash("terrain")){
			if(collision.type == collisionData::Type::onBegin){
				data->resetJump();
			}
		}
	}

};
