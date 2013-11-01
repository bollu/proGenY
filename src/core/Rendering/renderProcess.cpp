#pragma once
#include "renderProcess.h"

renderProcess::renderProcess(processMgr &processManager, Settings &settings, eventMgr &_eventManager) :
Process("renderProcess"){

	windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
	this->window = windowProc->getWindow();

	
};

void renderProcess::Draw(){

	for(renderProcess::baseRenderNode *node : nodes){
		//this->window->draw(node.drawable);
		node->Draw(this->window);
	}
};

void renderProcess::addRenderNode(renderProcess::baseRenderNode *node){
	this->nodes.push_back(node);
	this->nodes.sort(renderProcess::sortFn);	
};

void renderProcess::removeRenderNode(renderProcess::baseRenderNode *toRemove){
	this->nodes.remove(toRemove);
}



