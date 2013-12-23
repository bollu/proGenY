
#include "renderProcess.h"

renderProcess::renderProcess(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
Process("renderProcess"){

	windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
	this->window = windowProc->getWindow();

	
};

void renderProcess::Draw(){
	for(_RenderNode *node : nodes){
		//if (node->disabled_) continue;
		drawShape_(*node);

		switch(node->type_) {
			case _RenderNode::Type::Sprite:
				drawSprite_(*node);
				break;

			case _RenderNode::Type::Text:
				drawText_(*node);
				break;

			case _RenderNode::Type::Shape:
				drawShape_(*node);
				break;
		}
	}
};

void renderProcess::addRenderNode(_RenderNode &node){
	this->nodes.sort(renderProcess::sortFn);
	this->nodes.push_back(&node);
};

void renderProcess::removeRenderNode(_RenderNode &toRemove){
	this->nodes.remove(&toRemove);
}


void renderProcess::drawSprite_(_RenderNode &node){
	window->draw(*node.renderer_.sprite);
};

void renderProcess::drawShape_(_RenderNode &node){
	window->draw(*node.renderer_.shape);
};

void renderProcess::drawText_(_RenderNode &node){
	window->draw(*node.renderer_.text);

};



void setRenderNodePosition(_RenderNode &node, vector2 position){
	switch(node.type_) {
		case _RenderNode::Type::Sprite:
			node.renderer_.sprite->setPosition(position);
			break;

		case _RenderNode::Type::Text:
			node.renderer_.text->setPosition(position);
			break;

		case _RenderNode::Type::Shape:
			node.renderer_.shape->setPosition(position);
			break;
	}
};

void setRenderNodeAngle(_RenderNode &node, util::Angle angle){
	switch(node.type_) {
		case _RenderNode::Type::Sprite:
			node.renderer_.sprite->setRotation(angle.toDeg());
			break;

		case _RenderNode::Type::Text:
			node.renderer_.text->setRotation(angle.toDeg());
			break;

		case _RenderNode::Type::Shape:
			node.renderer_.shape->setRotation(angle.toDeg());
			break;
	}
};


void freeRenderNode(_RenderNode &node){
	switch(node.type_) {
		case _RenderNode::Type::Sprite:
			delete node.renderer_.sprite;
			break;

		case _RenderNode::Type::Text:
			delete node.renderer_.text;
			break;

		case _RenderNode::Type::Shape:
			delete node.renderer_.shape;
			break;
	}
};
