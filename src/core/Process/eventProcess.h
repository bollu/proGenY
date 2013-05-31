#pragma once
#include "Process.h"
#include "processMgr.h"
#include "../include/SFML/Graphics.hpp"
#include "windowProcess.h"

/*
 Events generated by this class:
 
 WindowClosed:
 *generated when the window is closed
 *data: NULL
 

*/
 class eventProcess : public Process{
 	sf::RenderWindow *window;
 	sf::Event event;

 	eventMgr &eventManager;

 	void _handleEvent();

 	void _handleWindowCloseEvent();

 	void _handleMouseButtonPressed();
 	void _handleMouseButtonReleased();

 	void _handleMouseMove();

 	void _handleKeyboardPressed();
 	void _handleKeyboardReleased();


 public:
 	eventProcess(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
 	Process("eventProcess"), eventManager(_eventManager){

 		windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));

 		this->window = windowProc->getWindow();
 	}

 	void preUpdate();

};