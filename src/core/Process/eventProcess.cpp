#pragma once
#include "eventProcess.h"
#include "../Property.h"


void eventProcess::preUpdate(){

	while(window->pollEvent(event)){
		this->_handleEvent();
	}
};


void eventProcess::_handleEvent(){
	switch(this->event.type){
		case sf::Event::Closed:
		this->_handleWindowCloseEvent();
		break;

		case sf::Event::MouseButtonPressed:
		this->_handleMouseButtonPressed();
		break;

		case sf::Event::MouseButtonReleased:
		this->_handleMouseButtonReleased();
		break;

		case sf::Event::MouseMoved:
		this->_handleMouseMove();
		break;

		case sf::Event::KeyPressed:
		this->_handleKeyboardPressed();
		break;

		case sf::Event::KeyReleased:
		this->_handleKeyboardReleased();
		break;

		case sf::Event::MouseWheelMoved:
		this->_handleMouseWheelMove();
		break;
	}
};

void eventProcess::_handleWindowCloseEvent(){
	static const Hash *windowClosed = Hash::getHash("windowClosed"); 

	eventManager.sendEvent(windowClosed);
//	window->close(); //<- HACK for now
};


void eventProcess::_handleMouseButtonPressed(){
	static const Hash *mouseLeftPressed = Hash::getHash("mouseLeftPressedScreen"); 
	static const Hash *mouseRightPressed = Hash::getHash("mouseRightPressedScreen"); 
	

	
	vector2 mousePos = vector2(event.mouseButton.x, event.mouseButton.y);
	
	if(event.mouseButton.button == sf::Mouse::Button::Left){
		eventManager.sendEvent(mouseLeftPressed, mousePos);
	}
	else if(event.mouseButton.button == sf::Mouse::Button::Right){
		eventManager.sendEvent(mouseRightPressed, mousePos);
	}

};
void eventProcess::_handleMouseButtonReleased(){
	static const Hash *mouseLeftReleased = Hash::getHash("mouseLeftReleasedScreen"); 
	static const Hash *mouseRightReleased = Hash::getHash("mouseRightReleasedScreen"); 

	vector2 mousePos = vector2(event.mouseButton.x, event.mouseButton.y);
	
	if(event.mouseButton.button == sf::Mouse::Button::Left){
		eventManager.sendEvent(mouseLeftReleased, mousePos);
	}
	else if(event.mouseButton.button == sf::Mouse::Button::Right){
		eventManager.sendEvent(mouseRightReleased, mousePos);
	}
};

void eventProcess::_handleMouseMove(){
	static const Hash *mouseMoved = Hash::getHash("mouseMovedScreen"); 

	vector2 mousePos = vector2(event.mouseMove.x, event.mouseMove.y);
	
	eventManager.sendEvent(mouseMoved, mousePos);

};

void eventProcess::_handleKeyboardPressed(){
	static const Hash *keyPressed = Hash::getHash("keyPressed"); 

	sf::Event::KeyEvent keyEvent = event.key;
	
	eventManager.sendEvent(keyPressed, keyEvent);
};
void eventProcess::_handleKeyboardReleased(){
	static const Hash *keyReleased = Hash::getHash("keyReleased"); 

	sf::Event::KeyEvent keyEvent = event.key;
	
	eventManager.sendEvent(keyReleased, keyEvent);
};


void eventProcess::_handleMouseWheelMove(){
	static const Hash *mouseWheelUp = Hash::getHash("mouseWheelUp"); 
	static const Hash *mouseWheelDown = Hash::getHash("mouseWheelDown"); 

	sf::Event::MouseWheelEvent mouseWheelEvent = event.mouseWheel;

	int delta = event.mouseWheel.delta;

	if(delta > 0){
		eventManager.sendEvent<int>(mouseWheelUp, delta);
	}else{
		delta = -delta;
		eventManager.sendEvent<int>(mouseWheelDown, delta);
	}
};
