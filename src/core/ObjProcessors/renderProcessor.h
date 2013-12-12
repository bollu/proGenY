#pragma once
#include "../objectProcessor.h"
#include "../include/SFML/Graphics.hpp"
#include "../vector.h"
#include <vector>
#include "../../util/strHelper.h"

#include "../Process/processMgr.h"
#include "../Settings.h"
#include "../Messaging/eventMgr.h"
#include "../Process/viewProcess.h"
#include "../Process/renderProcess.h"




/*!The data that is used by the renderProcessor to render objects
It's a collection of Renderer objects. as one Object may like to render
multiple things, the RenderData class acts as a "bag" to hold multiple Renderer
objects

\sa renderProcesor Renderer
*/
class RenderData{
private:
	friend class renderProcessor;
	std::vector< renderProcess::baseRenderNode *>renderers;
public:
	
	/*! Add a Renderer the RenderData */
	void addRenderer(renderProcess::baseRenderNode *renderer){
		this->renderers.push_back(renderer);
	}

	//whether the renderer should be shifted by half.
	//useful when manually setting positions / screwing around
	bool centered;

	RenderData() : centered(false){};
};

/*! an objectProcessor that handles rendering Object
the Object must be attached with RenderData. the objectProcessor
uses RenderData to draw the Object
*/
class renderProcessor : public objectProcessor{
private:
	sf::RenderWindow *window;
	viewProcess *view;
	renderProcess *render;

	void _Render(vector2 pos, util::Angle &angle , RenderData *data, bool centered);
public:

	renderProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager);
	void onObjectAdd(Object *obj);

	void Process(float dt);
	void onObjectRemove(Object *obj);
};
