#pragma once
#include "Process.h"
#include "processMgr.h"
#include "../include/SFML/Graphics.hpp"



class windowProcess : public Process{
	sf::RenderWindow *window;
public:
	windowProcess(processMgr &processManager, Settings &settings, eventMgr &eventManager) : 
	Process("windowProcess"){
		
		vector2 *screenDim = settings.getProp<vector2>(Hash::getHash("screenDimensions"));

		this->window = new sf::RenderWindow(sf::VideoMode(screenDim->x, screenDim->y), "ProGenY");
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