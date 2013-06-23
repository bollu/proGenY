#pragma once
#include "WSADHandler.h"

void WSADHandler::recieveEvent(const Hash *eventName, baseProperty *eventData){
	static const Hash *keyPressedHash = Hash::getHash("keyPressed");
	static const Hash *keyReleasedHash = Hash::getHash("keyReleased");
	static const Hash *mouseMovedGameHash = Hash::getHash("mouseMovedGame");


	if(eventName == mouseMovedGameHash){
		
		v2Prop *mousePos = dynamic_cast<v2Prop *>(eventData); 
		assert(mousePos != NULL);
			
		this->_updateGunFacing(*mousePos->getVal());
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

	if(key == this->WSADData.fireGun){
		this->_fireGun();
		
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


	phyData *phy = WSADData.physicsData;

	
	for(collisionData collision : phy->collisions){
		if(collision.data->collisionType == Hash::getHash("terrain")){
			if(collision.type == collisionData::Type::onBegin){
				data->resetJump();
			}
		}
	}

};

void WSADHandler::_updateGunFacing(vector2 gameMousePos){
	 

	
	vector2 delta = (gameMousePos - *this->WSADData.playerPos).Normalize();
	util::Angle facing = util::Angle(delta);
	
	float rad = 1;
	vector2 offset = facing.polarProjection(3);

	WSADData._gunData->setBulletPos(*WSADData.playerPos + offset);
	WSADData._gunData->setFacing(facing);

	WSADData.gunOffsetData->posOffset = offset;
	WSADData.gunOffsetData->angleOffset = facing;
};

void WSADHandler::_fireGun(){

	WSADData._gunData->Fire();
	
};
