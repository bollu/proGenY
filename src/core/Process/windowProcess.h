#pragma once
#include "Process.h"
#include "processMgr.h"
#include "../include/SFML/Graphics.hpp"



class windowProcess : public Process{
	sf::RenderWindow *window;
public:
	windowProcess(processMgr &processManager, Settings &settings, eventMgr &eventManager) : 
	Process("windowProcess"){
		this->window = new sf::RenderWindow(sf::VideoMode(1280, 720), "ProGenY");
	}

	virtual void preUpdate(){
		window->clear();
	};
	
	virtual void postDraw(){
		window->display();
	}

	sf::RenderWindow *getWindow(){
		return this->window;
	}

};