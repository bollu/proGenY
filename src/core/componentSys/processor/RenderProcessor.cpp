#pragma once
#include "RenderProcessor.h"

RenderProcessor::RenderProcessor(processMgr &processManager, 
	Settings &settings, EventManager &_eventManager) : ObjectProcessor("renderProcesor"){

	this->window = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"))->getWindow();
	this->view = processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));
	this->render = processManager.getProcess<renderProcess>(Hash::getHash("renderProcess"));

	windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
	windowProc->setClearColor(sf::Color(255, 255, 255, 255));

};


RenderData RenderProcessor::createRenderData(_RenderNode *renderNodes, int numRenderNodes){
	assert(numRenderNodes > 0 && renderNodes != NULL);
	RenderData renderData;
	renderData.renderNodes = new _RenderNode[numRenderNodes];

	for(int i = 0; i < numRenderNodes; i++){
		renderData.renderNodes[i] = renderNodes[i];
	}
	
	renderData.numRenderNodes = numRenderNodes;

	return renderData;

};



void RenderProcessor::_onObjectActivate(Object *obj){
	RenderData *data = obj->getPrimitive<RenderData>(Hash::getHash("RenderData"));
	assert(data != NULL);
	
	assert(data->numRenderNodes > 0 && data->renderNodes != NULL);
	for(int i = 0; i < data->numRenderNodes; i++) {
		this->render->addRenderNode(data->renderNodes[i]);
	}
};
void RenderProcessor::_onObjectDeactivate(Object *obj){
	RenderData *data = obj->getPrimitive<RenderData>(Hash::getHash("RenderData"));
	
	for(int i = 0; i < data->numRenderNodes; i++){
		this->render->removeRenderNode(data->renderNodes[i]);
	}
};


void RenderProcessor::_onObjectDeath(Object *obj){
	RenderData *data = obj->getPrimitive<RenderData>(Hash::getHash("RenderData"));
	if(data == NULL){
		return;
	}

	for(int i = 0; i < data->numRenderNodes; i++){
		this->render->removeRenderNode(data->renderNodes[i]);
		freeRenderNode(data->renderNodes[i]);
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

	//HACK! - disabled off center
	//this->_Render(renderPos, gameAngle, data, data->centered);
	this->_Render(renderPos, gameAngle, data, true);
}


void RenderProcessor::_Render(vector2 pos, util::Angle &angle, RenderData *data,  bool centered){

	//loop through the renderers
	for(int i = 0; i < data->numRenderNodes; i++){
		_RenderNode &renderer = data->renderNodes[i];
		setRenderNodePosition(renderer, pos);
		setRenderNodeAngle(renderer, angle);
	}
};
