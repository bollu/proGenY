
#include "renderProcess.h"

renderProcess::renderProcess(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
Process("renderProcess"){

	windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
	this->window = windowProc->getWindow();

	
};

void renderProcess::Update(float dt) {
	this->dt_ = dt;
};

void renderProcess::Draw(){
	for(RenderNode *node : nodes){
		switch(node->type_) {
			case RenderNode::Type::Sprite:
				drawSprite_(*node);
				break;

			case RenderNode::Type::Text:
				drawText_(*node);
				break;

			case RenderNode::Type::Shape:
				drawShape_(*node);
				break;

			case RenderNode::Type::Particle:
				drawParticleSystem_(*node);
				break;

			default:
				assert(false);
		}
	}
};

void renderProcess::addRenderNode(RenderNode &node){
	this->nodes.sort(renderProcess::sortFn);
	this->nodes.push_back(&node);
};

void renderProcess::removeRenderNode(RenderNode &toRemove){
	this->nodes.remove(&toRemove);
}


void renderProcess::drawSprite_(RenderNode &node){
	window->draw(*node.renderer_.sprite);
};

void renderProcess::drawShape_(RenderNode &node){
	window->draw(*node.renderer_.shape);
};

void renderProcess::drawText_(RenderNode &node){
	window->draw(*node.renderer_.text);

};

void renderProcess::drawParticleSystem_(RenderNode &node) {
	ParticleSystem &particleSystem = *node.renderer_.particleSystem;
	particleSystem.Update(this->dt_);
	particleSystem.Draw(*window);
	//window->draw(particleSystem);
};


void setRenderNodePosition(RenderNode &node, vector2 position){
	switch(node.type_) {
		case RenderNode::Type::Sprite:
			node.renderer_.sprite->setPosition(position);
			break;

		case RenderNode::Type::Text:
			node.renderer_.text->setPosition(position);
			break;

		case RenderNode::Type::Shape:
			node.renderer_.shape->setPosition(position);
			break;

		case RenderNode::Type::Particle:
			node.renderer_.particleSystem->setPosition(position);
			break;
		
		default:
			assert(false);
	}
};

void setRenderNodeAngle(RenderNode &node, util::Angle angle){
	switch(node.type_) {
		case RenderNode::Type::Sprite:
			node.renderer_.sprite->setRotation(angle.toDeg());
			break;

		case RenderNode::Type::Text:
			node.renderer_.text->setRotation(angle.toDeg());
			break;

		case RenderNode::Type::Shape:
			node.renderer_.shape->setRotation(angle.toDeg());
			break;

		case RenderNode::Type::Particle:
			node.renderer_.particleSystem->setRotation(angle.toDeg());
			break;
		
		default:
			assert(false);
	}
};


void freeRenderNode(RenderNode &node){
	switch(node.type_) {
		case RenderNode::Type::Sprite:
			delete node.renderer_.sprite;
			break;

		case RenderNode::Type::Text:
			delete node.renderer_.text;
			break;

		case RenderNode::Type::Shape:
			delete node.renderer_.shape;
			break;

		case RenderNode::Type::Particle:
			delete node.renderer_.particleSystem;
			break;

		default:
			assert(false);
	}
};
