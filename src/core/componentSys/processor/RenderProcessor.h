#pragma once
#include "ObjectProcessor.h"
#include "../include/SFML/Graphics.hpp"
#include "../../math/vector.h"
#include <vector>
#include "../../IO/strHelper.h"

#include "../../controlFlow/processMgr.h"
#include "../../IO/Settings.h"
#include "../../controlFlow/EventManager.h"
#include "../../Rendering/viewProcess.h"
#include "../../Rendering/renderProcess.h"


struct RenderData {
	RenderNode *renderNodes;
	int numRenderNodes;

private:
	RenderData() {
		numRenderNodes = 0;
		renderNodes = NULL;
	};
	friend class RenderProcessor;
};

/*! an ObjectProcessor that handles rendering Object
the Object must be attached with RenderData. the ObjectProcessor
uses RenderData to draw the Object
*/
class RenderProcessor : public ObjectProcessor{
public:

	RenderProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager);

	static RenderData createRenderData(RenderNode *renderNodes, int numRenderNodes = 1);
private:
	sf::RenderWindow *window;
	viewProcess *view;
	renderProcess *render;

	void _Render(vector2 pos, util::Angle &angle , RenderData *data, bool centered);

protected:
	void _onObjectActivate(Object *obj);
	void _onObjectDeactivate(Object *obj);
	
	void _Process(Object *obj, float dt);
	void _onObjectDeath(Object *obj);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("RenderData");
	};
};
