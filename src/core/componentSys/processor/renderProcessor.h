#pragma once
#include "objectProcessor.h"
#include "../include/SFML/Graphics.hpp"
#include "../../math/vector.h"
#include <vector>
#include "../../IO/strHelper.h"

#include "../../controlFlow/processMgr.h"
#include "../../IO/Settings.h"
#include "../../controlFlow/eventMgr.h"
#include "../../Rendering/viewProcess.h"
#include "../../Rendering/renderProcess.h"




/*!The data that is used by the renderProcessor to render objects
It's a collection of Renderer objects. as one Object may like to render
multiple things, the renderData class acts as a "bag" to hold multiple Renderer
objects

\sa renderProcesor Renderer
*/
class renderData{
private:
	friend class renderProcessor;
	std::vector< renderProcess::baseRenderNode *>renderers;
public:
	
	/*! Add a Renderer the renderData */
	void addRenderer(renderProcess::baseRenderNode *renderer){
		this->renderers.push_back(renderer);
	}

	//whether the renderer should be shifted by half.
	//useful when manually setting positions / screwing around
	bool centered;

	renderData() : centered(false){};
};

/*! an objectProcessor that handles rendering Object
the Object must be attached with renderData. the objectProcessor
uses renderData to draw the Object
*/
class renderProcessor : public objectProcessor{
public:

	renderProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager);

private:
	sf::RenderWindow *window;
	viewProcess *view;
	renderProcess *render;

	void _Render(vector2 pos, util::Angle &angle , renderData *data, bool centered);

protected:
	void _onObjectAdd(Object *obj);

	void _onObjectActivate(Object *obj);
	void _onObjectDeactivate(Object *obj);
	
	void _Process(Object *obj, float dt);
	void _onObjectDeath(Object *obj);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("renderData");
	};
};
