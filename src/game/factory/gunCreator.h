#pragma once
#include "objectFactory.h"
#include "../../core/Rendering/viewProcess.h"
#include "../ObjProcessors/gunProcessor.h"
#include "../ObjProcessors/offsetProcessor.h"

#include "../../core/componentSys/processor/renderProcessor.h"
#include "../../core/componentSys/processor/phyProcessor.h"
#include "../../core/Rendering/renderUtil.h"

#include "../defines/renderingLayers.h"


class gunCreator : public objectCreator{
private:
	viewProcess *viewProc;
	const Hash *enemyCollision;

	Object *parent;
	gunData gun;

	float radius;
public:

	gunCreator(viewProcess *_viewProc) : viewProc(_viewProc), parent(NULL), radius(0){}

	void Init(Object *parent, gunData gun, float offsetRadius = 0){
		this->parent = parent;
		this->gun = gun;
		this->radius = 0;
	}

	Object *createObject(vector2 gunPos) const{
		renderData render;
		offsetData offset;
		
		Object *gun = new Object("gun");

		vector2 *pos = gun->getPrimitive<vector2>(Hash::getHash("position"));
		*pos = gunPos;


		//renderer------------------------------------
		vector2 gunDim = vector2(2,1) * viewProc->getGame2RenderScale();
		sf::Shape *shape = renderUtil::createRectangleShape(gunDim);

		sf::Color color;
		color.r = rand() % 255;
		color.g = rand() % 255;
		color.b = rand() % 255;
		shape->setFillColor(color);
		//shape->setOutlineColor(sf::Color::White);
		//shape->setOutlineThickness(-2.0);

		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::aboveAction);
		render.addRenderer(renderer);
		//render.centered = true;
		
		//offset-------------------------------------
		assert(this->parent != NULL);
		offset.parent = this->parent;
		offset.offsetAngle = false;
		

		//final---------------------------------
		gun->addProp(Hash::getHash("renderData"), 
			new Prop<renderData>(render));
		gun->addProp(Hash::getHash("gunData"), 
			new Prop<gunData>(this->gun));
		gun->addProp(Hash::getHash("offsetData"), 
			new Prop<offsetData>(offset));
		
		return gun;
	};
};
