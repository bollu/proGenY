#pragma once

#pragma once
#include "Process.h"
#include "processMgr.h"
#include "../include/SFML/Graphics.hpp"

#include "windowProcess.h"

/*!

*/

/*
	Game Coord - box2d coordinates
	Render Coord - box2d coordinates * scaling
	Screen Coord - box2d coordinates * scaling + inverted


*/
 
class viewProcess : public Process{
	sf::RenderWindow *window;
	float windowHeight;

	sf::View defaultView;

	float game2RenderScale;

public:
	viewProcess(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
	 Process("viewProcess"){
		
	 	windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
		this->window = windowProc->getWindow();
		
		vector2 windowDim = vector2::cast<sf::Vector2u>(this->window->getSize());
		this->windowHeight = windowDim.y;

		this->game2RenderScale = 26;

		vector2 center = windowDim * 0.5;
	
	
		defaultView.setCenter(center.x, center.y);
		defaultView.setSize(windowDim.x, windowDim.y);
		
	}

	void Update(float dt){
		window->setView(defaultView);
	};
	
	
	vector2 game2RenderCoord(vector2 gameCoord){
		return gameCoord * game2RenderScale;
	};

	vector2 render2GameCoord(vector2 renderCoord){
		return renderCoord * (1.0 / game2RenderScale);
	};

	vector2 render2ScreenCoord(vector2 renderCoord){
		return vector2(renderCoord.x, this->windowHeight - renderCoord.y);
	}


	vector2 screen2RenderCoord(vector2 screenCoord){
		return vector2(screenCoord.x, this->windowHeight - screenCoord.y);
	}

	vector2 game2ScreenCoord(vector2 gameCoord){
		return this->render2ScreenCoord(this->game2RenderCoord(gameCoord));
	}

	vector2 screen2GameCoord(vector2 screenCoord){
		return this->render2GameCoord(screen2RenderCoord(screenCoord));
	}

	void move(vector2 offset){
		this->defaultView.move(offset.x, offset.y);
	}

	void setCenter(vector2 center){
		this->defaultView.setCenter(center.x, center.y);
	}

	vector2 getCenter(){
		return vector2::cast(this->defaultView.getCenter());
	}

	float getGame2RenderScale(){
		return this->game2RenderScale;
	}

	float getRender2GameScale(){
		return 1.0f / this->game2RenderScale;
	}
	
};