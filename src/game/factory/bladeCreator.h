#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../ObjProcessors/bladeProcessor.h"
#include "../ObjProcessors/offsetProcessor.h"

#include "../../core/ObjProcessors/renderProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../../core/renderUtil.h"

#include "../defines/renderingLayers.h"


class bladeCreator : public objectCreator{
private:
	viewProcess *viewProc;
	const Hash *enemyCollision;

	Object *parent;
	bladeData blade;

	float radius;
public:

	bladeCreator(viewProcess *_viewProc) : viewProc(_viewProc), parent(NULL), radius(0){}

	void setParent(Object *parent){
		this->parent = parent;
	}

	void setGunData(bladeData blade){
		this->blade = blade;
	}


	Object *createObject(vector2 pos) const{
		renderData render;
		offsetData offset;
		
		Object *blade = new Object("blade");

		blade->setProp<vector2>(Hash::getHash("position"), pos);


		//renderer------------------------------------
		vector2 bladeDim = vector2(5 ,1) * viewProc->getGame2RenderScale();
		sf::Shape *shape = renderUtil::createRectangleShape(bladeDim);
		shape->setFillColor(sf::Color::White);
		shape->setOutlineColor(sf::Color::Black);
		shape->setOutlineThickness(-3.0);

		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::aboveAction);
		render.addRenderer(renderer);
	
		//offset-------------------------------------
		assert(this->parent != NULL);
		offset.parent = this->parent;
		offset.offsetAngle = false;
		

		//final---------------------------------
		blade->addProp(Hash::getHash("renderData"), 
			new Prop<renderData>(render));
		blade->addProp(Hash::getHash("bladeData"), 
			new Prop<bladeData>(this->blade));
		blade->addProp(Hash::getHash("offsetData"), 
			new Prop<offsetData>(offset));
		
		return blade;
	};
};