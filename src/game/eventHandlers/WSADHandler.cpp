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
	util::msgLog("key pressed");

	sf::Keyboard::Key key = event->code;

	if(key == this->WSADData.up){
		WSADData.objMoveData->jump = true;
	}

	if(key == this->WSADData.left){
		WSADData.objMoveData->moveLeft = true;
		
	}

	if(key == this->WSADData.right){
		WSADData.objMoveData->moveRight = true;
		
	}
};
void WSADHandler::_handleKeyRelease(sf::Event::KeyEvent *event){
	util::msgLog("key released");

	sf::Keyboard::Key key = event->code;

	if(key == this->WSADData.up){
	
	}

	if(key == this->WSADData.left){
		WSADData.objMoveData->moveLeft = false;


	}

	if(key == this->WSADData.right){
		WSADData.objMoveData->moveRight = false;
	}
};

void WSADHandler::Update(){


	moveData *data = WSADData.objMoveData;
	//you're not moving left or right
	if(!data->moveLeft && !data->moveRight ){
		data->stopMoving = true;
		
	}else{
		data->stopMoving = false;
	}
};
