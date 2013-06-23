#pragma once
#include "viewProcess.h"

viewProcess::viewProcess(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
 Process("viewProcess"), eventManager(_eventManager){
	
 	windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
	this->window = windowProc->getWindow();
	
	vector2 windowDim = vector2::cast<sf::Vector2u>(this->window->getSize());
	this->windowHeight = windowDim.y;

	this->game2RenderScale = 26;
	vector2 center = windowDim * 0.5;


	defaultView.setCenter(center.x, center.y);
	defaultView.setSize(windowDim.x, windowDim.y);
	
	_eventManager.Register(Hash::getHash("mouseMovedScreen"), this);
}

void viewProcess::Update(float dt){
	window->setView(defaultView);
};


vector2 viewProcess::game2ViewCoord(vector2 gameCoord){
	return gameCoord * game2RenderScale;
};

vector2 viewProcess::view2GameCoord(vector2 renderCoord){
	return renderCoord * (1.0 / game2RenderScale);
};

vector2 viewProcess::view2RenderCoord(vector2 renderCoord){
	return vector2(renderCoord.x, this->windowHeight - renderCoord.y);
}


vector2 viewProcess::render2ViewCoord(vector2 screenCoord){
	return vector2(screenCoord.x, this->windowHeight - screenCoord.y);
}

vector2 viewProcess::screen2RenderCoord(vector2 screenCoord){
	sf::Vector2f renderCoord = window->mapPixelToCoords(vector2::cast<sf::Vector2i>(screenCoord));
	return vector2::cast(renderCoord);
};
 
vector2 viewProcess::render2ScreeenCoord(vector2 renderCoord){
	sf::Vector2i screenCoord = window->mapCoordsToPixel(vector2::cast<sf::Vector2f>(renderCoord));
	return vector2::cast(renderCoord);
};



void viewProcess::move(vector2 offset){
	this->defaultView.move(offset.x, offset.y);
}

void viewProcess::setCenter(vector2 center){
	this->defaultView.setCenter(center.x, center.y);
}

void viewProcess::setRotation(util::Angle angle){
	this->defaultView.setRotation(angle.toDeg());
}
vector2 viewProcess::getCenter(){
	return vector2::cast(this->defaultView.getCenter());
}

float viewProcess::getGame2RenderScale(){
	return this->game2RenderScale;
}

float viewProcess::getRender2GameScale(){
	return 1.0f / this->game2RenderScale;
}

void viewProcess::recieveEvent(const Hash *eventName, baseProperty *eventData){

	/*auto keyEventProp = (Prop<sf::Event::KeyEvent> *)(eventData);

	sf::Event::KeyEvent *keyEvent = keyEventProp->getVal();
	*/
	if(eventName == Hash::getHash("mouseMovedScreen")){

		v2Prop *mousePosProp = dynamic_cast<v2Prop *>(eventData);
		assert(mousePosProp != NULL);
		vector2 *mousePos = mousePosProp->getVal();
		
		vector2 renderMousePos = screen2RenderCoord(*mousePos);
		vector2 gameMousePos = view2GameCoord(render2ViewCoord(renderMousePos));

		eventManager.sendEvent(Hash::getHash("mouseMovedGame"), gameMousePos); 
		

	//	PRINTVECTOR2(gameMousePos);
	}
}