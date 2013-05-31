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
	}
};

void eventProcess::_handleWindowCloseEvent(){
	static const Hash *windowClosed = Hash::getHash("windowClosed"); 

	eventManager.sendEvent(windowClosed);
	window->close(); //<- HACK for now
};


void eventProcess::_handleMouseButtonPressed(){
	static const Hash *mouseLeftPressed = Hash::getHash("mouseLeftPressed"); 
	static const Hash *mouseRightPressed = Hash::getHash("mouseRightPressed"); 
	

	
	v2Prop mousePos = v2Prop(vector2(event.mouseButton.x, event.mouseButton.y));

	if(event.mouseButton.button == sf::Mouse::Button::Left){
		eventManager.sendEvent(mouseLeftPressed, &mousePos);
	}
	else if(event.mouseButton.button == sf::Mouse::Button::Right){
		eventManager.sendEvent(mouseRightPressed, &mousePos);
	}

};
void eventProcess::_handleMouseButtonReleased(){
	static const Hash *mouseLeftReleased = Hash::getHash("mouseLeftReleased"); 
	static const Hash *mouseRightReleased = Hash::getHash("mouseRightReleased"); 

	v2Prop mousePos = v2Prop(vector2(event.mouseButton.x, event.mouseButton.y));

	if(event.mouseButton.button == sf::Mouse::Button::Left){
		eventManager.sendEvent(mouseLeftReleased, &mousePos);
	}
	else if(event.mouseButton.button == sf::Mouse::Button::Right){
		eventManager.sendEvent(mouseRightReleased, &mousePos);
	}
};

void eventProcess::_handleMouseMove(){
	static const Hash *mouseMoved = Hash::getHash("mouseMoved"); 

	v2Prop mousePos = v2Prop(vector2(event.mouseMove.x, event.mouseMove.y));

	eventManager.sendEvent(mouseMoved, &mousePos);

};

void eventProcess::_handleKeyboardPressed(){
	static const Hash *keyPressed = Hash::getHash("keyPressed"); 

	sf::Event::KeyEvent keyEvent = event.key;
	auto keyEventProp = Prop<sf::Event::KeyEvent>(event.key);

	eventManager.sendEvent(keyPressed, &keyEventProp);
};
void eventProcess::_handleKeyboardReleased(){
	static const Hash *keyReleased = Hash::getHash("keyReleased"); 

	sf::Event::KeyEvent keyEvent = event.key;
	auto keyEventProp = Prop<sf::Event::KeyEvent>(event.key);

	eventManager.sendEvent(keyReleased, &keyEventProp);
};
