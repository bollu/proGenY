#pragma once
#include "renderProcessor.h"

renderProcessor::renderProcessor(processMgr &processManager, 
	Settings &settings, eventMgr &_eventManager){

	this->window = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"))->getWindow();
	this->view = processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));
	this->render = processManager.getProcess<renderProcess>(Hash::getHash("renderProcess"));

	windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
	windowProc->setClearColor(sf::Color(255, 255, 255, 255));

};


void renderProcessor::onObjectAdd(Object *obj){
	RenderData *data = obj->getProp<RenderData>(Hash::getHash("RenderData"));
	if(data == NULL){
		return;
	}

	for(renderProcess::baseRenderNode *node : data->renderers){
		this->render->addRenderNode(node);
	}
}

void renderProcessor::onObjectRemove(Object *obj){
	RenderData *data = obj->getProp<RenderData>(Hash::getHash("RenderData"));
	if(data == NULL){
		return;
	}


	for(renderProcess::baseRenderNode *node : data->renderers){
		this->render->removeRenderNode(node);
		//delete(node);
	}
};

void renderProcessor::Process(float dt){

	for(auto it= objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;

		RenderData *data = obj->getProp<RenderData>(Hash::getHash("RenderData"));
		if(data == NULL){
			continue;
		}

		//for box2d, +ve x axis is 0 degree clockwise
		//ffor SFML, -ve y axis 0 degree. weird...
		//box2dClockwise = 360 - box2d
		static util::Angle game2RenderAngle = util::Angle::Deg(360);

		vector2* gamePos = obj->getProp<vector2>(Hash::getHash("position"));
		vector2 viewPos = view->game2ViewCoord(*gamePos);
		vector2 renderPos = view->view2RenderCoord(viewPos);

		util::Angle *angle = obj->getProp<util::Angle>(Hash::getHash("facing"));
		float fAngle = angle->toDeg();
		util::Angle gameAngle = game2RenderAngle - *angle;
		

		this->_Render(renderPos, gameAngle, data, data->centered);
	};
}


void renderProcessor::_Render(vector2 pos, util::Angle &angle, 
	RenderData *data,  bool centered){
		//loop through the renderers
	for(renderProcess::baseRenderNode *renderer : data->renderers){
		 	
		 	renderer->setPosition(pos);
		 	renderer->setRotation(angle);
	}
};