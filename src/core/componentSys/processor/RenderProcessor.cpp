#pragma once
#include "RenderProcessor.h"

RenderProcessor::RenderProcessor(processMgr &processManager, 
	Settings &settings, eventMgr &_eventManager) : ObjectProcessor("renderProcesor"){

	this->window = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"))->getWindow();
	this->view = processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));
	this->render = processManager.getProcess<renderProcess>(Hash::getHash("renderProcess"));

	windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
	windowProc->setClearColor(sf::Color(255, 255, 255, 255));

};


void RenderProcessor::_onObjectAdd(Object *obj){
	RenderData *data = obj->getPrimitive<RenderData>(Hash::getHash("RenderData"));
	if(data == NULL){
		return;
	}

	for(renderProcess::baseRenderNode *node : data->renderers){
		this->render->addRenderNode(node);
	}
}

void RenderProcessor::_onObjectActivate(Object *obj){
	RenderData *data = obj->getPrimitive<RenderData>(Hash::getHash("RenderData"));
	assert(data != NULL);

	for(renderProcess::baseRenderNode *node : data->renderers){
		this->render->addRenderNode(node);
	}

};
void RenderProcessor::_onObjectDeactivate(Object *obj){
	RenderData *data = obj->getPrimitive<RenderData>(Hash::getHash("RenderData"));
	assert(data != NULL);
	
	for(renderProcess::baseRenderNode *node : data->renderers){
		this->render->removeRenderNode(node);
	}
};


void RenderProcessor::_onObjectDeath(Object *obj){
	RenderData *data = obj->getPrimitive<RenderData>(Hash::getHash("RenderData"));
	if(data == NULL){
		return;
	}


	for(renderProcess::baseRenderNode *node : data->renderers){
		this->render->removeRenderNode(node);
		delete(node);
	}
};

void RenderProcessor::_Process(Object *obj, float dt){
	RenderData *data = obj->getPrimitive<RenderData>(Hash::getHash("RenderData"));

	//for box2d, +ve x axis is 0 degree clockwise
	//ffor SFML, -ve y axis 0 degree. weird...
	//box2dClockwise = 360 - box2d
	static util::Angle game2RenderAngle = util::Angle::Deg(360);

	vector2* gamePos = obj->getPrimitive<vector2>(Hash::getHash("position"));
	vector2 viewPos = view->game2ViewCoord(*gamePos);
	vector2 renderPos = view->view2RenderCoord(viewPos);

	util::Angle *angle = obj->getPrimitive<util::Angle>(Hash::getHash("facing"));
	float fAngle = angle->toDeg();
	util::Angle gameAngle = game2RenderAngle - *angle;


	this->_Render(renderPos, gameAngle, data, data->centered);
}


void RenderProcessor::_Render(vector2 pos, util::Angle &angle, 
	RenderData *data,  bool centered){
		//loop through the renderers
	for(renderProcess::baseRenderNode *renderer : data->renderers){

		renderer->setPosition(pos);
		renderer->setRotation(angle);
	}
};
